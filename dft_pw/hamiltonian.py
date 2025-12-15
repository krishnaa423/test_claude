"""
Hamiltonian construction for plane wave DFT.

The Kohn-Sham Hamiltonian in plane wave basis:
H = T + V_loc + V_nl + V_H + V_xc

where:
- T: Kinetic energy (diagonal)
- V_loc: Local pseudopotential
- V_nl: Nonlocal pseudopotential (separable form)
- V_H: Hartree potential
- V_xc: Exchange-correlation potential
"""

import numpy as np
from scipy import linalg
from typing import Dict, List, Tuple, Optional
from .crystal import Crystal
from .basis import PlaneWaveBasis, FFTGrid
from .pseudopotential import Pseudopotential, spherical_bessel_j
from .xc import get_xc_functional


class Hamiltonian:
    """
    Kohn-Sham Hamiltonian in plane wave basis.
    """

    def __init__(self, crystal: Crystal, pw_basis: PlaneWaveBasis,
                 fft_grid: FFTGrid, pseudopotentials: Dict[str, Pseudopotential],
                 xc_functional: str = 'LDA'):
        """
        Initialize Hamiltonian.

        Args:
            crystal: Crystal structure
            pw_basis: Plane wave basis (for specific k-point)
            fft_grid: FFT grid for real-space operations
            pseudopotentials: Dict mapping element symbol to Pseudopotential
            xc_functional: Name of XC functional
        """
        self.crystal = crystal
        self.pw_basis = pw_basis
        self.fft_grid = fft_grid
        self.pps = pseudopotentials
        self.xc = get_xc_functional(xc_functional)

        self.npw = pw_basis.npw

        # Precompute pseudopotential form factors
        self._setup_local_pp()
        self._setup_nonlocal_pp()

    def _setup_local_pp(self):
        """Precompute local pseudopotential in G-space."""
        G = self.pw_basis.g_vectors
        G_mag = np.sqrt(np.sum(G**2, axis=1))

        # Local potential for each species
        vloc_g = np.zeros(self.npw, dtype=np.complex128)

        for atom in self.crystal.atoms:
            pp = self.pps[atom.symbol]
            tau = atom.position @ self.crystal.cell  # Cartesian position

            # Get V_loc(G) for this species
            vloc_species = pp.get_vloc_of_g(G_mag, self.crystal.volume)

            # Structure factor
            sf = np.exp(-1j * G @ tau)

            vloc_g += vloc_species * sf

        self.vloc_g = vloc_g

    def _setup_nonlocal_pp(self):
        """Precompute nonlocal pseudopotential projectors."""
        G = self.pw_basis.g_vectors
        k = self.pw_basis.k
        kpG = k + G
        kpG_mag = np.sqrt(np.sum(kpG**2, axis=1))

        # Store projectors for each atom and angular momentum
        self.beta_kg = []  # List of (beta_g, l, dij, structure_factor) for each projector

        for iatom, atom in enumerate(self.crystal.atoms):
            pp = self.pps[atom.symbol]
            tau = atom.position @ self.crystal.cell

            if pp.projectors is None or len(pp.projectors) == 0:
                continue

            for iproj, proj in enumerate(pp.projectors):
                # Compute beta(k+G)
                beta_g = pp.get_projector_of_g(iproj, kpG_mag)

                # Spherical harmonics for |k+G| direction
                # For l=0, Y_00 = 1/sqrt(4*pi)
                # For higher l, we need proper spherical harmonics

                # Get angular part - simplified for s and p orbitals
                ylm = self._compute_ylm(proj.l, kpG)

                # Structure factor
                sf = np.exp(-1j * kpG @ tau)

                # Get D_ij coefficient
                if pp.dij_matrix is not None:
                    dij = pp.dij_matrix[iproj, iproj]
                else:
                    dij = 1.0

                self.beta_kg.append({
                    'beta': beta_g * ylm * sf,
                    'l': proj.l,
                    'dij': dij,
                    'atom_idx': iatom
                })

    def _compute_ylm(self, l: int, k: np.ndarray) -> np.ndarray:
        """
        Compute real spherical harmonics Y_lm for a set of vectors.

        For simplicity, we sum over all m for each l (spherical average).
        This is appropriate for norm-conserving PPs in the Kleinman-Bylander form.

        Args:
            l: Angular momentum
            k: Vectors (nk, 3)

        Returns:
            Spherical harmonic values (nk,)
        """
        nk = len(k)
        k_mag = np.sqrt(np.sum(k**2, axis=1))
        k_mag = np.maximum(k_mag, 1e-10)  # Avoid division by zero

        if l == 0:
            # Y_00 = 1/sqrt(4*pi)
            return np.ones(nk) / np.sqrt(4 * np.pi)
        elif l == 1:
            # Sum of |Y_1m|^2 = 3/(4*pi), but we want just the factor
            # For KB form, we use the normalized projector
            # Y_1m contributions: Y_10 ~ z/r, Y_11 ~ x/r, Y_1-1 ~ y/r
            # We'll use a scalar factor (spherical average)
            return np.sqrt(3.0 / (4 * np.pi)) * np.ones(nk)
        elif l == 2:
            return np.sqrt(5.0 / (4 * np.pi)) * np.ones(nk)
        else:
            return np.sqrt((2 * l + 1) / (4 * np.pi)) * np.ones(nk)

    def get_local_potential(self, rho_g: np.ndarray = None) -> np.ndarray:
        """
        Get total local potential (V_loc + V_H + V_xc) in G-space.

        Args:
            rho_g: Electron density in G-space (FFT grid)
                   If None, returns only V_loc (pseudopotential)

        Returns:
            V_local(G-G') matrix elements
        """
        if rho_g is None:
            return self.vloc_g

        # Total local potential = V_loc + V_H + V_xc
        # We need to compute this and extract plane wave components

        # This will be computed in the apply_H method
        return self.vloc_g

    def get_hartree_potential_g(self, rho_g: np.ndarray) -> np.ndarray:
        """
        Compute Hartree potential in G-space.

        V_H(G) = 4*pi*rho(G) / G^2

        Args:
            rho_g: Density in G-space on FFT grid

        Returns:
            Hartree potential on FFT grid
        """
        G2 = self.fft_grid.g2_grid
        vh_g = np.zeros_like(rho_g)

        # Avoid G=0
        mask = G2 > 1e-10
        vh_g[mask] = 4 * np.pi * rho_g[mask] / G2[mask]
        vh_g[~mask] = 0.0  # G=0 contribution handled separately

        return vh_g

    def build_matrix(self, v_eff_r: np.ndarray) -> np.ndarray:
        """
        Build full Hamiltonian matrix.

        H_GG' = T_G * delta_GG' + V_loc(G-G') + V_nl_GG'

        Args:
            v_eff_r: Effective potential in real space (V_loc + V_H + V_xc)

        Returns:
            Hamiltonian matrix (npw x npw)
        """
        H = np.zeros((self.npw, self.npw), dtype=np.complex128)

        # Kinetic energy (diagonal)
        np.fill_diagonal(H, self.pw_basis.kinetic_energies)

        # Local potential: V(G-G')
        # Transform V_eff to G-space on FFT grid
        v_eff_g = self.fft_grid.to_reciprocal_space(v_eff_r)

        # For each pair of plane waves, get V(G-G')
        for ig in range(self.npw):
            miller_g = self.pw_basis.miller_indices[ig]
            for igp in range(self.npw):
                miller_gp = self.pw_basis.miller_indices[igp]
                # G - G'
                delta_miller = miller_g - miller_gp

                # Map to FFT grid index
                idx = np.zeros(3, dtype=int)
                ng = self.fft_grid.ng
                for i in range(3):
                    idx[i] = delta_miller[i] if delta_miller[i] >= 0 else delta_miller[i] + ng[i]
                    # Check if within grid
                    if idx[i] < 0 or idx[i] >= ng[i]:
                        continue

                linear_idx = idx[0] * ng[1] * ng[2] + idx[1] * ng[2] + idx[2]
                if linear_idx < len(v_eff_g):
                    H[ig, igp] += v_eff_g[linear_idx]

        # Nonlocal potential: V_nl = sum_i |beta_i><beta_i| * D_ii
        for proj_data in self.beta_kg:
            beta = proj_data['beta']
            dij = proj_data['dij']
            # |beta><beta| contribution
            H += dij * np.outer(beta, np.conj(beta))

        return H

    def apply_H(self, psi: np.ndarray, v_eff_r: np.ndarray) -> np.ndarray:
        """
        Apply Hamiltonian to wavefunction (matrix-free).

        H|psi> = T|psi> + V_loc|psi> + V_nl|psi>

        Args:
            psi: Wavefunction in PW basis (npw,)
            v_eff_r: Effective potential in real space

        Returns:
            H|psi> (npw,)
        """
        # Kinetic energy (diagonal)
        H_psi = self.pw_basis.kinetic_energies * psi

        # Local potential: V * psi in real space, then back to G-space
        psi_g_grid = self.fft_grid.pw_to_grid(psi, self.pw_basis)
        psi_r = self.fft_grid.to_real_space(psi_g_grid)
        v_psi_r = v_eff_r * psi_r
        v_psi_g_grid = self.fft_grid.to_reciprocal_space(v_psi_r)
        v_psi_pw = self.fft_grid.grid_to_pw(v_psi_g_grid, self.pw_basis)
        H_psi += v_psi_pw

        # Nonlocal potential
        for proj_data in self.beta_kg:
            beta = proj_data['beta']
            dij = proj_data['dij']
            # <beta|psi>
            beta_psi = np.vdot(beta, psi)
            # D * |beta><beta|psi>
            H_psi += dij * beta * beta_psi

        return H_psi

    def diagonalize(self, v_eff_r: np.ndarray,
                    n_bands: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize Hamiltonian to get eigenvalues and eigenvectors.

        Args:
            v_eff_r: Effective potential in real space
            n_bands: Number of bands to compute (default: all)

        Returns:
            (eigenvalues, eigenvectors) where eigenvectors[:,i] is the i-th state
        """
        if n_bands is None:
            n_bands = self.npw

        # Build full matrix
        H = self.build_matrix(v_eff_r)

        # Diagonalize
        if n_bands < self.npw // 2:
            # Use sparse solver for fewer bands
            from scipy.sparse.linalg import eigsh
            from scipy.sparse import csr_matrix

            H_sparse = csr_matrix(H)
            eigenvalues, eigenvectors = eigsh(H_sparse, k=n_bands, which='SA')
            # Sort by eigenvalue
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        else:
            # Full diagonalization
            eigenvalues, eigenvectors = linalg.eigh(H)
            eigenvalues = eigenvalues[:n_bands]
            eigenvectors = eigenvectors[:, :n_bands]

        return eigenvalues, eigenvectors
