"""
Plane wave basis set module.
"""

import numpy as np
from typing import Tuple, Optional
from .crystal import Crystal


class PlaneWaveBasis:
    """
    Plane wave basis set for DFT calculations.

    The basis functions are plane waves: exp(i(k+G).r) / sqrt(V)
    where G are reciprocal lattice vectors satisfying |k+G|^2/2 < E_cut
    """

    def __init__(self, crystal: Crystal, ecut: float, k: np.ndarray = None):
        """
        Initialize plane wave basis for a given k-point.

        Args:
            crystal: Crystal structure
            ecut: Plane wave cutoff energy (in Hartree)
            k: k-point in Cartesian coordinates (Bohr^-1)
        """
        self.crystal = crystal
        self.ecut = ecut
        self.k = k if k is not None else np.zeros(3)

        self._generate_g_vectors()

    def _generate_g_vectors(self):
        """Generate G-vectors satisfying the cutoff criterion."""
        # Maximum G-vector components
        b = self.crystal.reciprocal_cell

        # Estimate maximum Miller indices needed
        # |G|^2/2 < ecut => |G| < sqrt(2*ecut)
        gmax = np.sqrt(2 * self.ecut)

        # Get approximate number of grid points in each direction
        b_lengths = np.linalg.norm(b, axis=1)
        nmax = np.ceil(gmax / b_lengths).astype(int) + 1

        # Generate all possible Miller indices
        miller_indices = []
        g_vectors = []
        kinetic_energies = []

        for n1 in range(-nmax[0], nmax[0] + 1):
            for n2 in range(-nmax[1], nmax[1] + 1):
                for n3 in range(-nmax[2], nmax[2] + 1):
                    n = np.array([n1, n2, n3])
                    G = n @ b  # G-vector in Cartesian
                    kpG = self.k + G
                    ekin = 0.5 * np.dot(kpG, kpG)

                    if ekin <= self.ecut:
                        miller_indices.append(n)
                        g_vectors.append(G)
                        kinetic_energies.append(ekin)

        # Sort by kinetic energy
        indices = np.argsort(kinetic_energies)
        self.miller_indices = np.array([miller_indices[i] for i in indices])
        self.g_vectors = np.array([g_vectors[i] for i in indices])
        self.kinetic_energies = np.array([kinetic_energies[i] for i in indices])
        self.npw = len(self.g_vectors)

    def get_kinetic_matrix(self) -> np.ndarray:
        """
        Get kinetic energy matrix (diagonal).

        Returns:
            Diagonal kinetic energy matrix
        """
        return np.diag(self.kinetic_energies)

    def get_kinetic_diagonal(self) -> np.ndarray:
        """
        Get kinetic energy diagonal.

        Returns:
            Kinetic energies for each G-vector
        """
        return self.kinetic_energies.copy()

    @property
    def size(self) -> int:
        """Number of plane waves in the basis."""
        return self.npw


class FFTGrid:
    """
    FFT grid for real-space / reciprocal-space transformations.
    """

    def __init__(self, crystal: Crystal, ecut: float, grid_factor: float = 2.0):
        """
        Initialize FFT grid.

        Args:
            crystal: Crystal structure
            ecut: Cutoff energy (Hartree)
            grid_factor: Factor for FFT grid density (default 2.0 for proper aliasing)
        """
        self.crystal = crystal
        self.ecut = ecut
        self.grid_factor = grid_factor

        self._setup_grid()

    def _setup_grid(self):
        """Set up FFT grid dimensions."""
        # Maximum G-vector magnitude
        gmax = np.sqrt(2 * self.ecut * self.grid_factor)

        # Grid spacing in reciprocal space
        b = self.crystal.reciprocal_cell
        b_lengths = np.linalg.norm(b, axis=1)

        # Number of grid points (must be even for efficiency)
        self.ng = np.ceil(gmax / b_lengths * 2).astype(int)
        # Make grid sizes even for FFT efficiency
        self.ng = np.array([n + (n % 2) for n in self.ng])

        self.nr = self.ng.copy()  # Real space grid same as reciprocal
        self.nrtot = np.prod(self.nr)

        # Set up G-vector indexing for FFT
        self._setup_fft_indices()

    def _setup_fft_indices(self):
        """Set up FFT indices for G-vectors."""
        b = self.crystal.reciprocal_cell

        # Generate all G-vectors on the FFT grid
        g_list = []
        miller_list = []

        for n1 in range(self.ng[0]):
            i1 = n1 if n1 < self.ng[0] // 2 else n1 - self.ng[0]
            for n2 in range(self.ng[1]):
                i2 = n2 if n2 < self.ng[1] // 2 else n2 - self.ng[1]
                for n3 in range(self.ng[2]):
                    i3 = n3 if n3 < self.ng[2] // 2 else n3 - self.ng[2]
                    miller = np.array([i1, i2, i3])
                    G = miller @ b
                    g_list.append(G)
                    miller_list.append(miller)

        self.g_grid = np.array(g_list)
        self.miller_grid = np.array(miller_list)
        self.g2_grid = np.sum(self.g_grid**2, axis=1)

    def to_real_space(self, f_g: np.ndarray) -> np.ndarray:
        """
        Transform from G-space to real space using FFT.

        Args:
            f_g: Function on FFT G-grid (ng1*ng2*ng3,)

        Returns:
            Function in real space (nr1*nr2*nr3,)
        """
        f_g_3d = f_g.reshape(self.ng)
        f_r_3d = np.fft.ifftn(f_g_3d) * self.nrtot
        return f_r_3d.flatten()

    def to_reciprocal_space(self, f_r: np.ndarray) -> np.ndarray:
        """
        Transform from real space to G-space using FFT.

        Args:
            f_r: Function in real space (nr1*nr2*nr3,)

        Returns:
            Function on FFT G-grid (ng1*ng2*ng3,)
        """
        f_r_3d = f_r.reshape(self.nr)
        f_g_3d = np.fft.fftn(f_r_3d) / self.nrtot
        return f_g_3d.flatten()

    def pw_to_grid(self, coeffs: np.ndarray, pw_basis: PlaneWaveBasis) -> np.ndarray:
        """
        Map plane wave coefficients to FFT grid.

        Args:
            coeffs: Plane wave coefficients (npw,)
            pw_basis: Plane wave basis

        Returns:
            Coefficients on FFT grid (ng1*ng2*ng3,)
        """
        f_g = np.zeros(np.prod(self.ng), dtype=np.complex128)

        for ipw, miller in enumerate(pw_basis.miller_indices):
            # Map Miller indices to FFT grid indices
            idx = np.zeros(3, dtype=int)
            for i in range(3):
                idx[i] = miller[i] if miller[i] >= 0 else miller[i] + self.ng[i]

            # Linear index
            linear_idx = idx[0] * self.ng[1] * self.ng[2] + idx[1] * self.ng[2] + idx[2]
            f_g[linear_idx] = coeffs[ipw]

        return f_g

    def grid_to_pw(self, f_g: np.ndarray, pw_basis: PlaneWaveBasis) -> np.ndarray:
        """
        Extract plane wave coefficients from FFT grid.

        Args:
            f_g: Coefficients on FFT grid
            pw_basis: Plane wave basis

        Returns:
            Plane wave coefficients (npw,)
        """
        coeffs = np.zeros(pw_basis.npw, dtype=np.complex128)

        for ipw, miller in enumerate(pw_basis.miller_indices):
            idx = np.zeros(3, dtype=int)
            for i in range(3):
                idx[i] = miller[i] if miller[i] >= 0 else miller[i] + self.ng[i]

            linear_idx = idx[0] * self.ng[1] * self.ng[2] + idx[1] * self.ng[2] + idx[2]
            coeffs[ipw] = f_g[linear_idx]

        return coeffs

    def get_density_from_wfcs(self, wfcs: np.ndarray, occupations: np.ndarray,
                              pw_basis: PlaneWaveBasis) -> np.ndarray:
        """
        Compute charge density from wavefunctions.

        Args:
            wfcs: Wavefunctions (npw, nbands)
            occupations: Occupation numbers (nbands,)
            pw_basis: Plane wave basis

        Returns:
            Charge density on real-space grid
        """
        rho_r = np.zeros(self.nrtot)

        for iband in range(wfcs.shape[1]):
            if occupations[iband] > 1e-10:
                # Map to FFT grid
                psi_g = self.pw_to_grid(wfcs[:, iband], pw_basis)
                # Transform to real space
                psi_r = self.to_real_space(psi_g)
                # Add |psi|^2 weighted by occupation
                rho_r += occupations[iband] * np.abs(psi_r)**2

        # Normalize
        rho_r /= self.crystal.volume

        return rho_r
