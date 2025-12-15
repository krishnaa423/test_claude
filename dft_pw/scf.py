"""
Self-Consistent Field (SCF) solver for DFT calculations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .crystal import Crystal
from .basis import PlaneWaveBasis, FFTGrid
from .kpoints import KPoints
from .hamiltonian import Hamiltonian
from .pseudopotential import Pseudopotential
from .xc import get_xc_functional


@dataclass
class SCFResult:
    """Results from SCF calculation."""
    converged: bool
    n_iterations: int
    total_energy: float
    eigenvalues: Dict[int, np.ndarray]  # k-point index -> eigenvalues
    occupations: Dict[int, np.ndarray]  # k-point index -> occupations
    density: np.ndarray  # Final density on real-space grid
    fermi_energy: float


class DensityMixer:
    """
    Density mixing schemes for SCF convergence.
    """

    def __init__(self, method: str = 'linear', alpha: float = 0.3,
                 n_history: int = 5):
        """
        Initialize mixer.

        Args:
            method: 'linear', 'pulay', or 'kerker'
            alpha: Mixing parameter
            n_history: History length for Pulay mixing
        """
        self.method = method
        self.alpha = alpha
        self.n_history = n_history

        # History for Pulay mixing
        self.rho_history = []
        self.residual_history = []

    def mix(self, rho_in: np.ndarray, rho_out: np.ndarray) -> np.ndarray:
        """
        Mix input and output densities.

        Args:
            rho_in: Input density
            rho_out: Output density from diagonalization

        Returns:
            Mixed density for next iteration
        """
        if self.method == 'linear':
            return self._linear_mix(rho_in, rho_out)
        elif self.method == 'pulay':
            return self._pulay_mix(rho_in, rho_out)
        else:
            return self._linear_mix(rho_in, rho_out)

    def _linear_mix(self, rho_in: np.ndarray, rho_out: np.ndarray) -> np.ndarray:
        """Simple linear mixing."""
        return (1 - self.alpha) * rho_in + self.alpha * rho_out

    def _pulay_mix(self, rho_in: np.ndarray, rho_out: np.ndarray) -> np.ndarray:
        """Pulay (DIIS) mixing."""
        residual = rho_out - rho_in

        self.rho_history.append(rho_in.copy())
        self.residual_history.append(residual.copy())

        # Limit history
        if len(self.rho_history) > self.n_history:
            self.rho_history.pop(0)
            self.residual_history.pop(0)

        n = len(self.rho_history)
        if n < 2:
            return self._linear_mix(rho_in, rho_out)

        # Build overlap matrix
        B = np.zeros((n + 1, n + 1))
        for i in range(n):
            for j in range(n):
                B[i, j] = np.dot(self.residual_history[i], self.residual_history[j])
        B[n, :n] = 1.0
        B[:n, n] = 1.0
        B[n, n] = 0.0

        # Solve for coefficients
        rhs = np.zeros(n + 1)
        rhs[n] = 1.0

        try:
            coeffs = np.linalg.solve(B, rhs)
        except np.linalg.LinAlgError:
            return self._linear_mix(rho_in, rho_out)

        # Build mixed density
        rho_mixed = np.zeros_like(rho_in)
        for i in range(n):
            rho_mixed += coeffs[i] * (self.rho_history[i] + self.alpha * self.residual_history[i])

        return rho_mixed

    def reset(self):
        """Reset mixer history."""
        self.rho_history = []
        self.residual_history = []


class SCFSolver:
    """
    Self-Consistent Field solver.
    """

    def __init__(self, crystal: Crystal, kpoints: KPoints,
                 ecut: float, pseudopotentials: Dict[str, Pseudopotential],
                 xc_functional: str = 'LDA', n_bands: int = None,
                 smearing: float = 0.01, smearing_type: str = 'fermi-dirac'):
        """
        Initialize SCF solver.

        Args:
            crystal: Crystal structure
            kpoints: K-point grid
            ecut: Plane wave cutoff (Hartree)
            pseudopotentials: Dict of pseudopotentials
            xc_functional: XC functional name
            n_bands: Number of bands (default: n_electrons/2 + 4)
            smearing: Smearing width (Hartree)
            smearing_type: 'fermi-dirac' or 'gaussian'
        """
        self.crystal = crystal
        self.kpoints = kpoints
        self.ecut = ecut
        self.pps = pseudopotentials
        self.xc_name = xc_functional
        self.smearing = smearing
        self.smearing_type = smearing_type

        # Determine number of valence electrons from pseudopotentials
        self.n_electrons = sum(
            self.pps[atom.symbol].z_valence for atom in crystal.atoms
        )

        # Number of bands
        if n_bands is None:
            n_bands = int(self.n_electrons / 2) + 4
        self.n_bands = n_bands

        # Set up FFT grid (shared for all k-points)
        self.fft_grid = FFTGrid(crystal, ecut)

        # XC functional
        self.xc = get_xc_functional(xc_functional)

        # Set up basis and Hamiltonian for each k-point
        self.pw_bases = {}
        self.hamiltonians = {}

        for ik, (k_cart, weight) in enumerate(kpoints):
            pw_basis = PlaneWaveBasis(crystal, ecut, k_cart)
            self.pw_bases[ik] = pw_basis
            self.hamiltonians[ik] = Hamiltonian(
                crystal, pw_basis, self.fft_grid, pseudopotentials, xc_functional
            )

    def solve(self, max_iter: int = 100, tol: float = 1e-6,
              mixing: str = 'pulay', mixing_alpha: float = 0.3,
              verbose: bool = True) -> SCFResult:
        """
        Run SCF calculation.

        Args:
            max_iter: Maximum iterations
            tol: Convergence tolerance on energy
            mixing: Density mixing method
            mixing_alpha: Mixing parameter
            verbose: Print progress

        Returns:
            SCFResult with converged quantities
        """
        mixer = DensityMixer(method=mixing, alpha=mixing_alpha)

        # Initialize density (superposition of atomic densities)
        rho_r = self._initialize_density()

        # Storage for eigenvalues and wavefunctions
        eigenvalues = {}
        eigenvectors = {}
        occupations = {}

        prev_energy = 0.0
        converged = False

        for iteration in range(max_iter):
            # Compute effective potential
            v_eff_r = self._compute_effective_potential(rho_r)

            # Diagonalize at each k-point
            for ik in range(self.kpoints.nkpts):
                eigs, evecs = self.hamiltonians[ik].diagonalize(
                    v_eff_r, n_bands=self.n_bands
                )
                eigenvalues[ik] = eigs
                eigenvectors[ik] = evecs

            # Compute Fermi energy and occupations
            fermi_energy, occupations = self._compute_occupations(eigenvalues)

            # Compute new density from wavefunctions
            rho_out = self._compute_density(eigenvectors, occupations)

            # Mix densities
            rho_r = mixer.mix(rho_r, rho_out)

            # Ensure density is positive and normalized
            rho_r = np.maximum(rho_r, 1e-20)
            # Normalize
            dv = self.crystal.volume / self.fft_grid.nrtot
            total_charge = np.sum(rho_r) * dv
            rho_r *= self.n_electrons / total_charge

            # Compute total energy
            total_energy = self._compute_total_energy(
                rho_r, eigenvalues, occupations, v_eff_r
            )

            # Check convergence
            energy_diff = abs(total_energy - prev_energy)

            if verbose:
                print(f"SCF iter {iteration + 1:3d}: E = {total_energy:16.8f} Ha, "
                      f"dE = {energy_diff:12.2e}, Ef = {fermi_energy:8.4f} Ha")

            if energy_diff < tol and iteration > 0:
                converged = True
                if verbose:
                    print(f"\nSCF converged in {iteration + 1} iterations!")
                break

            prev_energy = total_energy

        if not converged and verbose:
            print(f"\nWarning: SCF did not converge in {max_iter} iterations")

        return SCFResult(
            converged=converged,
            n_iterations=iteration + 1,
            total_energy=total_energy,
            eigenvalues=eigenvalues,
            occupations=occupations,
            density=rho_r,
            fermi_energy=fermi_energy
        )

    def _initialize_density(self) -> np.ndarray:
        """Initialize density from superposition of atomic densities."""
        rho_r = np.zeros(self.fft_grid.nrtot)

        # Simple initialization: uniform density
        rho_r[:] = self.n_electrons / self.crystal.volume

        return rho_r

    def _compute_effective_potential(self, rho_r: np.ndarray) -> np.ndarray:
        """
        Compute effective potential V_eff = V_loc + V_H + V_xc.

        Args:
            rho_r: Density in real space

        Returns:
            Effective potential in real space
        """
        # Transform density to G-space
        rho_g = self.fft_grid.to_reciprocal_space(rho_r)

        # Hartree potential
        vh_g = np.zeros_like(rho_g)
        G2 = self.fft_grid.g2_grid
        mask = G2 > 1e-10
        vh_g[mask] = 4 * np.pi * rho_g[mask] / G2[mask]
        vh_g[~mask] = 0.0

        vh_r = self.fft_grid.to_real_space(vh_g).real

        # XC potential
        vxc_r = self.xc.get_vxc_potential(rho_r)

        # Local pseudopotential
        # Sum over atoms
        vloc_g = np.zeros(len(G2), dtype=np.complex128)
        G = self.fft_grid.g_grid
        G_mag = np.sqrt(G2)

        for atom in self.crystal.atoms:
            pp = self.pps[atom.symbol]
            tau = atom.position @ self.crystal.cell

            # V_loc(G)
            vloc_species = pp.get_vloc_of_g(G_mag, self.crystal.volume)

            # Structure factor
            sf = np.exp(-1j * G @ tau)
            vloc_g += vloc_species * sf

        vloc_r = self.fft_grid.to_real_space(vloc_g).real

        return vloc_r + vh_r + vxc_r

    def _compute_occupations(self, eigenvalues: Dict[int, np.ndarray]
                             ) -> Tuple[float, Dict[int, np.ndarray]]:
        """
        Compute Fermi energy and occupation numbers.

        Uses Fermi-Dirac or Gaussian smearing.

        Args:
            eigenvalues: Dict of eigenvalues per k-point

        Returns:
            (fermi_energy, occupations_dict)
        """
        # Collect all eigenvalues with k-point weights
        all_eigs = []
        all_weights = []

        for ik, (_, weight) in enumerate(self.kpoints):
            eigs = eigenvalues[ik]
            for e in eigs:
                all_eigs.append(e)
                all_weights.append(weight)

        all_eigs = np.array(all_eigs)
        all_weights = np.array(all_weights)

        # Find Fermi energy by bisection
        e_min, e_max = np.min(all_eigs) - 1.0, np.max(all_eigs) + 1.0

        def count_electrons(ef):
            total = 0.0
            for ik, (_, weight) in enumerate(self.kpoints):
                eigs = eigenvalues[ik]
                occs = self._fermi_function(eigs, ef)
                total += 2.0 * weight * np.sum(occs)  # Factor of 2 for spin
            return total

        # Bisection to find Fermi energy
        for _ in range(100):
            e_mid = 0.5 * (e_min + e_max)
            n_mid = count_electrons(e_mid)
            if n_mid < self.n_electrons:
                e_min = e_mid
            else:
                e_max = e_mid
            if abs(e_max - e_min) < 1e-10:
                break

        fermi_energy = e_mid

        # Compute occupations
        occupations = {}
        for ik, (_, weight) in enumerate(self.kpoints):
            eigs = eigenvalues[ik]
            occs = 2.0 * self._fermi_function(eigs, fermi_energy)  # Factor of 2 for spin
            occupations[ik] = occs

        return fermi_energy, occupations

    def _fermi_function(self, energies: np.ndarray, ef: float) -> np.ndarray:
        """Fermi-Dirac distribution function."""
        if self.smearing < 1e-10:
            return np.where(energies <= ef, 1.0, 0.0)

        x = (energies - ef) / self.smearing

        if self.smearing_type == 'fermi-dirac':
            # Avoid overflow
            x = np.clip(x, -50, 50)
            return 1.0 / (1.0 + np.exp(x))
        else:  # gaussian
            from scipy.special import erfc
            return 0.5 * erfc(x)

    def _compute_density(self, eigenvectors: Dict[int, np.ndarray],
                         occupations: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Compute electron density from wavefunctions.

        rho(r) = sum_k sum_n f_nk |psi_nk(r)|^2
        """
        rho_r = np.zeros(self.fft_grid.nrtot)

        for ik, (_, weight) in enumerate(self.kpoints):
            pw_basis = self.pw_bases[ik]
            evecs = eigenvectors[ik]
            occs = occupations[ik]

            for iband in range(evecs.shape[1]):
                if occs[iband] < 1e-10:
                    continue

                # Get wavefunction in G-space
                psi_pw = evecs[:, iband]

                # Map to FFT grid
                psi_g = self.fft_grid.pw_to_grid(psi_pw, pw_basis)

                # Transform to real space
                psi_r = self.fft_grid.to_real_space(psi_g)

                # Add |psi|^2 weighted by occupation and k-weight
                rho_r += weight * occs[iband] * np.abs(psi_r)**2

        # Normalize by volume
        rho_r /= self.crystal.volume

        return rho_r

    def _compute_total_energy(self, rho_r: np.ndarray,
                              eigenvalues: Dict[int, np.ndarray],
                              occupations: Dict[int, np.ndarray],
                              v_eff_r: np.ndarray) -> float:
        """
        Compute total energy.

        E_tot = sum_k sum_n f_nk * e_nk - E_H + E_xc - int[V_xc * rho]
        """
        dv = self.crystal.volume / self.fft_grid.nrtot

        # Band energy
        e_band = 0.0
        for ik, (_, weight) in enumerate(self.kpoints):
            eigs = eigenvalues[ik]
            occs = occupations[ik]
            e_band += weight * np.sum(occs * eigs)

        # Hartree energy (double counting correction)
        rho_g = self.fft_grid.to_reciprocal_space(rho_r)
        G2 = self.fft_grid.g2_grid

        e_hartree = 0.0
        mask = G2 > 1e-10
        e_hartree = 2 * np.pi * np.sum(np.abs(rho_g[mask])**2 / G2[mask]).real
        e_hartree *= self.crystal.volume

        # XC energy
        e_xc = self.xc.get_exc_energy(rho_r, self.crystal.volume)

        # XC potential contribution (to subtract double counting)
        vxc_r = self.xc.get_vxc_potential(rho_r)
        e_vxc = np.sum(vxc_r * rho_r) * dv

        # Total energy
        e_total = e_band - e_hartree + e_xc - e_vxc

        return e_total
