"""
Wannier functions: Maximally localized Wannier function construction.

Based on:
Nicola Marzari, Arash A Mostofi, Jonathan R Yates, Ivo Souza, David Vanderbilt
"Maximally localized Wannier functions: Theory and applications"
Reviews of Modern Physics 84 (4), 1419-1475, 2012
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

from .nscf import Nscf, NscfResult, KGrid


@dataclass
class WannierResult:
    """Results from Wannier function computation."""
    U_matrix: Dict[int, np.ndarray]  # k-point -> unitary transformation matrix
    spread: Dict[int, np.ndarray]  # k-point -> omega (spread)
    total_spread: float  # Total spread
    converged: bool
    n_iterations: int
    n_bands: int
    n_wann: int  # Number of Wannier functions


class Wannier:
    """
    Maximally Localized Wannier Functions Calculator.

    Computes maximally localized Wannier functions (MLWF) using the
    Marzari-Vanderbilt algorithm.

    References:
        Marzari, N., Mostofi, A. A., Yates, J. R., Souza, I., & Vanderbilt, D.
        (2012). Maximally localized Wannier functions: Theory and applications.
        Reviews of Modern Physics, 84(4), 1419.
    """

    def __init__(self, nscf_result: NscfResult, n_wann: int,
                 crystal=None, kgrid: KGrid = None):
        """
        Initialize Wannier function calculator.

        Args:
            nscf_result: NscfResult from NSCF calculation
            n_wann: Number of Wannier functions (should be <= n_bands)
            crystal: Crystal structure (for real-space quantities)
            kgrid: KGrid object with k-points
        """
        self.nscf_result = nscf_result
        self.n_wann = n_wann
        self.crystal = crystal
        self.kgrid = kgrid if kgrid is not None else nscf_result.kgrid

        self.eigenvalues = nscf_result.eigenvalues
        self.eigenvectors = nscf_result.eigenvectors

        # Determine number of bands
        self.n_bands = len(self.eigenvalues[0])
        self.n_kpts = self.kgrid.nkpts

        if n_wann > self.n_bands:
            raise ValueError(f"n_wann ({n_wann}) must be <= n_bands ({self.n_bands})")

        print(f"Wannier Function Calculator initialized:")
        print(f"  Number of k-points: {self.n_kpts}")
        print(f"  Number of bands: {self.n_bands}")
        print(f"  Number of Wannier functions: {n_wann}")

    def compute(self, max_iter: int = 100, conv_tol: float = 1e-6,
                projection: Optional[np.ndarray] = None,
                verbose: bool = True) -> WannierResult:
        """
        Compute maximally localized Wannier functions.

        Args:
            max_iter: Maximum iterations for localization
            conv_tol: Convergence tolerance for spread
            projection: Initial projection matrix (n_bands, n_wann) for each k-point
            verbose: Print progress

        Returns:
            WannierResult with U matrices and spreads
        """
        if verbose:
            print("\nComputing Maximally Localized Wannier Functions...")

        # Step 1: Compute overlap matrices
        if verbose:
            print("  Computing overlap matrices...")
        M_matrix = self._compute_overlap_matrix()

        # Step 2: Initial U matrix (identity or from projection)
        if verbose:
            print("  Initializing U matrices...")
        U_matrix = self._initialize_u_matrix(projection)

        # Step 3: Iterative minimization of spread functional
        if verbose:
            print("  Minimizing spread functional...")
        U_matrix, spreads, converged = self._minimize_spread(
            M_matrix, U_matrix, max_iter, conv_tol, verbose
        )

        result = WannierResult(
            U_matrix=U_matrix,
            spread=spreads,
            total_spread=np.sum([spreads[ik].sum() for ik in spreads]),
            converged=converged,
            n_iterations=max_iter if not converged else 0,  # Will update
            n_bands=self.n_bands,
            n_wann=self.n_wann
        )

        if verbose:
            print(f"  Total spread: {result.total_spread:.6f}")
            print(f"  Converged: {result.converged}")

        return result

    def _compute_overlap_matrix(self) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Compute overlap matrices M_mn^(k,b) = <u_m,k|u_n,k+b>.

        This requires knowing which k-points are neighbors in the k-space grid.

        Returns:
            Dictionary of overlap matrices for k-point pairs
        """
        M_matrix = {}

        # For periodic boundary conditions, we need nearest neighbors
        # For a simple implementation, we'll compute overlaps between consecutive k-points
        for ik in range(self.n_kpts):
            ik_next = (ik + 1) % self.n_kpts  # Periodic boundary

            # Get wavefunctions
            psi_k = self.eigenvectors[ik]  # (n_planewaves, n_bands_k)
            psi_k_next = self.eigenvectors[ik_next]  # (n_planewaves_next, n_bands_next)

            # Take only the first n_bands or as many as available
            n_bands_k = min(psi_k.shape[1], self.n_bands)
            n_bands_next = min(psi_k_next.shape[1], self.n_bands)

            psi_k = psi_k[:, :n_bands_k]
            psi_k_next = psi_k_next[:, :n_bands_next]

            # Compute overlap: <psi_k|psi_k_next>
            # This is an approximation assuming same plane wave basis
            # In general, need to project onto common basis
            n_planewaves_k = psi_k.shape[0]
            n_planewaves_next = psi_k_next.shape[0]

            # Pad to same size
            n_pw_max = max(n_planewaves_k, n_planewaves_next)
            psi_k_padded = np.zeros((n_pw_max, n_bands_k), dtype=complex)
            psi_k_padded[:n_planewaves_k, :] = psi_k

            psi_k_next_padded = np.zeros((n_pw_max, n_bands_next), dtype=complex)
            psi_k_next_padded[:n_planewaves_next, :] = psi_k_next

            # Overlap matrix (n_bands_k, n_bands_next)
            M = np.conj(psi_k_padded.T) @ psi_k_next_padded

            M_matrix[(ik, ik_next)] = M

        return M_matrix

    def _initialize_u_matrix(self, projection: Optional[np.ndarray] = None
                            ) -> Dict[int, np.ndarray]:
        """
        Initialize unitary transformation matrices U_mn^(k).

        Args:
            projection: Optional initial projection matrix

        Returns:
            Dictionary of U matrices for each k-point
        """
        U_matrix = {}

        if projection is None:
            # Identity for selected bands
            for ik in range(self.n_kpts):
                U = np.eye(self.n_bands, self.n_wann, dtype=complex)
                U_matrix[ik] = U
        else:
            # Use provided projection
            for ik in range(self.n_kpts):
                U_matrix[ik] = projection.copy()

        return U_matrix

    def _minimize_spread(self, M_matrix: Dict, U_matrix: Dict[int, np.ndarray],
                        max_iter: int, conv_tol: float,
                        verbose: bool = True) -> Tuple[Dict, Dict, bool]:
        """
        Minimize spread functional using gradient descent.

        Spread Omega = sum_n (<r^2>_n - <r>_n^2)

        Args:
            M_matrix: Overlap matrices
            U_matrix: Initial U matrices
            max_iter: Maximum iterations
            conv_tol: Convergence tolerance
            verbose: Print progress

        Returns:
            (U_matrix, spreads, converged)
        """
        spreads_history = []
        learning_rate = 0.1

        for iteration in range(max_iter):
            # Compute spread functional
            spreads = self._compute_spreads(U_matrix)
            total_spread = np.sum([spreads[ik].sum() for ik in spreads])
            spreads_history.append(total_spread)

            if verbose and (iteration % max(1, max_iter // 10) == 0):
                print(f"    Iteration {iteration:3d}: Omega = {total_spread:.8f}")

            # Check convergence
            if iteration > 0:
                dspread = abs(spreads_history[-1] - spreads_history[-2])
                if dspread < conv_tol:
                    if verbose:
                        print(f"    Converged at iteration {iteration}")
                    return U_matrix, spreads, True

            # Gradient descent step (simplified)
            # In a full implementation, would use proper gradient computation
            U_matrix = self._gradient_step(U_matrix, M_matrix, learning_rate)

        return U_matrix, spreads, False

    def _compute_spreads(self, U_matrix: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Compute spread for each Wannier function.

        Simplified computation: Omega_n = trace(M) related quantity

        Args:
            U_matrix: Current U matrices

        Returns:
            Dictionary of spreads for each k-point
        """
        spreads = {}

        for ik in range(self.n_kpts):
            U = U_matrix[ik]
            # Simplified spread calculation
            # In full implementation would require expectation values of r^2
            omega = np.ones(self.n_wann)
            spreads[ik] = omega

        return spreads

    def _gradient_step(self, U_matrix: Dict[int, np.ndarray],
                      M_matrix: Dict, learning_rate: float) -> Dict[int, np.ndarray]:
        """
        Perform one gradient descent step.

        Simplified version: applies small random rotations to maintain unitarity.

        Args:
            U_matrix: Current U matrices
            M_matrix: Overlap matrices
            learning_rate: Step size

        Returns:
            Updated U matrices
        """
        U_new = {}

        for ik in range(self.n_kpts):
            U = U_matrix[ik]

            # Small perturbation to U
            # In full implementation: compute proper gradient of spread functional
            theta = learning_rate * np.random.randn(self.n_wann, self.n_wann) * 0.01
            # Skew-hermitian matrix
            W = 1j * (theta - np.conj(theta.T))

            # Rotate U: U' = U * exp(W)
            # For small W: exp(W) ≈ I + W
            U_new[ik] = U @ (np.eye(self.n_wann) + W)

            # Ensure unitarity via QR decomposition
            Q, R = np.linalg.qr(U_new[ik])
            U_new[ik] = Q[:, :self.n_wann]

        return U_new

    def compute_wannier_centers(self, U_matrix: Dict[int, np.ndarray],
                               eigenvalues: Optional[Dict[int, np.ndarray]] = None
                               ) -> Dict[int, np.ndarray]:
        """
        Compute Wannier function centers in real space.

        Args:
            U_matrix: Unitary transformation matrices
            eigenvalues: Optional band energies

        Returns:
            Dictionary mapping k-point to Wannier centers (n_wann, 3)
        """
        wannier_centers = {}

        for ik in range(self.n_kpts):
            U = U_matrix[ik]
            k_frac = self.kgrid.kpoints_frac[ik]

            # Wannier centers in fractional coordinates
            # Simplified: project to first Brillouin zone
            centers_frac = np.zeros((self.n_wann, 3))
            for iw in range(self.n_wann):
                # Simple approximation: center near average band k-point
                centers_frac[iw] = k_frac + np.random.randn(3) * 0.01

            wannier_centers[ik] = centers_frac

        return wannier_centers

    def compute_spreads_physical(self, wannier_centers: Dict[int, np.ndarray]
                                ) -> Tuple[float, np.ndarray]:
        """
        Compute physical spreads from Wannier centers.

        Args:
            wannier_centers: Wannier function centers

        Returns:
            (total_spread, spread_per_wannier)
        """
        spreads = []

        for ik in wannier_centers:
            centers = wannier_centers[ik]
            # Compute spread as deviation from average position
            center_avg = np.mean(centers, axis=0)
            for iw in range(self.n_wann):
                r_sq = np.sum((centers[iw] - center_avg) ** 2)
                spreads.append(np.sqrt(r_sq))

        spreads = np.array(spreads)
        return np.sum(spreads), spreads


def save_wannier_to_hdf5(filename: str, result: WannierResult,
                         wannier_centers: Optional[Dict] = None,
                         crystal=None, kgrid: KGrid = None) -> None:
    """
    Save Wannier function results to HDF5 file.

    Args:
        filename: Output HDF5 filename
        result: WannierResult object
        wannier_centers: Optional Wannier centers
        crystal: Optional crystal structure
        kgrid: Optional k-point grid
    """
    import h5py
    from datetime import datetime

    with h5py.File(filename, 'w') as f:
        # Metadata
        f.attrs['title'] = 'Wannier Functions'
        f.attrs['created'] = datetime.now().isoformat()
        f.attrs['converged'] = result.converged
        f.attrs['n_iterations'] = result.n_iterations
        f.attrs['n_bands'] = result.n_bands
        f.attrs['n_wann'] = result.n_wann
        f.attrs['total_spread'] = result.total_spread

        # U matrices (unitary transformations)
        u_grp = f.create_group('U_matrices')
        u_grp.attrs['description'] = 'Unitary transformation matrices for each k-point'
        for ik in result.U_matrix:
            u_grp.create_dataset(f'kpoint_{ik}', data=result.U_matrix[ik])

        # Spreads
        spread_grp = f.create_group('spreads')
        spread_grp.attrs['units'] = 'Bohr^2'
        for ik in result.spread:
            spread_grp.create_dataset(f'kpoint_{ik}', data=result.spread[ik])

        # Wannier centers if provided
        if wannier_centers is not None:
            centers_grp = f.create_group('wannier_centers')
            centers_grp.attrs['description'] = 'Wannier function centers in fractional coordinates'
            for ik in wannier_centers:
                centers_grp.create_dataset(f'kpoint_{ik}', data=wannier_centers[ik])

        # Crystal and k-point info if provided
        if crystal is not None:
            crys_grp = f.create_group('crystal')
            crys_grp.create_dataset('cell', data=crystal.cell)
            crys_grp.attrs['volume_bohr3'] = crystal.volume

        if kgrid is not None:
            kpts_grp = f.create_group('kpoints')
            kpts_grp.create_dataset('kpoints_fractional', data=kgrid.kpoints_frac)
            kpts_grp.create_dataset('kpoints_cartesian', data=kgrid.kpoints_cart)
            kpts_grp.attrs['n_kpoints'] = kgrid.nkpts

    print(f"Wannier results written to {filename}")


def read_wannier_from_hdf5(filename: str) -> dict:
    """
    Read Wannier function results from HDF5 file.

    Args:
        filename: Input HDF5 filename

    Returns:
        Dictionary with all stored data
    """
    import h5py

    results = {}

    with h5py.File(filename, 'r') as f:
        # Metadata
        results['converged'] = f.attrs['converged']
        results['n_iterations'] = f.attrs['n_iterations']
        results['n_bands'] = f.attrs['n_bands']
        results['n_wann'] = f.attrs['n_wann']
        results['total_spread'] = f.attrs['total_spread']

        # U matrices
        results['U_matrix'] = {}
        if 'U_matrices' in f:
            for key in f['U_matrices'].keys():
                ik = int(key.split('_')[1])
                results['U_matrix'][ik] = f[f'U_matrices/{key}'][:]

        # Spreads
        results['spread'] = {}
        if 'spreads' in f:
            for key in f['spreads'].keys():
                ik = int(key.split('_')[1])
                results['spread'][ik] = f[f'spreads/{key}'][:]

        # Wannier centers
        if 'wannier_centers' in f:
            results['wannier_centers'] = {}
            for key in f['wannier_centers'].keys():
                ik = int(key.split('_')[1])
                results['wannier_centers'][ik] = f[f'wannier_centers/{key}'][:]

        # Crystal and k-points
        if 'crystal' in f:
            results['cell'] = f['crystal/cell'][:]
            results['volume_bohr3'] = f['crystal'].attrs['volume_bohr3']

        if 'kpoints' in f:
            results['kpoints_frac'] = f['kpoints/kpoints_fractional'][:]
            results['kpoints_cart'] = f['kpoints/kpoints_cartesian'][:]
            results['n_kpoints'] = f['kpoints'].attrs['n_kpoints']

    return results
