"""
Non-Self-Consistent Field (NSCF) calculations.

This module provides classes for computing eigenvalues and eigenvectors
on arbitrary k-points using a pre-computed charge density from SCF.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .crystal import Crystal
from .kpoints import KPoints
from .pseudopotential import Pseudopotential
from .scf import SCFSolver, SCFResult
from .hdf5_output import HDF5Output


@dataclass
class KGrid:
    """
    K-point grid in reduced coordinates.

    This class manages a set of k-points in fractional (reduced) coordinates
    with optional weights for integration.
    """

    kpoints_frac: np.ndarray  # Shape: (n_kpts, 3) in reduced coordinates
    weights: np.ndarray = None  # Shape: (n_kpts,), default: 1/n_kpts for each
    crystal: Crystal = None

    def __post_init__(self):
        """Initialize weights if not provided."""
        if self.weights is None:
            n_kpts = len(self.kpoints_frac)
            self.weights = np.ones(n_kpts) / n_kpts

        # Validate shapes
        assert len(self.kpoints_frac) == len(self.weights), \
            "Number of k-points must match number of weights"

    @property
    def nkpts(self) -> int:
        """Number of k-points."""
        return len(self.kpoints_frac)

    @property
    def kpoints_cart(self) -> np.ndarray:
        """Get k-points in Cartesian coordinates."""
        if self.crystal is None:
            raise ValueError("Crystal not set, cannot convert to Cartesian coordinates")
        return self.kpoints_frac @ self.crystal.reciprocal_cell

    @staticmethod
    def from_monkhorst_pack(crystal: Crystal, grid: Tuple[int, int, int],
                           shift: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> 'KGrid':
        """
        Create k-grid from Monkhorst-Pack mesh.

        Args:
            crystal: Crystal structure
            grid: Grid dimensions (nk1, nk2, nk3)
            shift: Grid shift in fractional coordinates

        Returns:
            KGrid object
        """
        nk1, nk2, nk3 = grid
        s1, s2, s3 = shift

        kpoints_frac = []
        for i1 in range(nk1):
            for i2 in range(nk2):
                for i3 in range(nk3):
                    k1 = (2 * i1 - nk1 + 1 + 2 * s1) / (2 * nk1)
                    k2 = (2 * i2 - nk2 + 1 + 2 * s2) / (2 * nk2)
                    k3 = (2 * i3 - nk3 + 1 + 2 * s3) / (2 * nk3)
                    kpoints_frac.append([k1, k2, k3])

        kpoints_frac = np.array(kpoints_frac)
        return KGrid(kpoints_frac, crystal=crystal)

    @staticmethod
    def from_path(crystal: Crystal, special_points: Dict[str, np.ndarray],
                  path_str: str, npoints_per_segment: int = 50) -> 'KGrid':
        """
        Create k-grid along a path through special points.

        Args:
            crystal: Crystal structure
            special_points: Dict mapping point names to fractional coordinates
            path_str: Path specification like 'GXWLGK'
            npoints_per_segment: Number of k-points per segment

        Returns:
            KGrid object
        """
        kpoints = []
        segment_start_idx = [0]

        for i, sym in enumerate(path_str):
            if sym not in special_points:
                raise ValueError(f"Special point {sym} not found in special_points")

            k_frac = special_points[sym]

            if i == 0:
                kpoints.append(k_frac)
            else:
                # Previous point
                prev_sym = path_str[i - 1]
                k_prev_frac = special_points[prev_sym]

                # Interpolate between previous and current point
                for j in range(1, npoints_per_segment + 1):
                    t = j / npoints_per_segment
                    k_interp = k_prev_frac + t * (k_frac - k_prev_frac)
                    kpoints.append(k_interp)

                segment_start_idx.append(len(kpoints) - 1)

        kpoints_frac = np.array(kpoints)
        # No weights for band structure paths
        weights = np.zeros(len(kpoints_frac))

        kg = KGrid(kpoints_frac, weights=weights, crystal=crystal)
        kg.segment_start_idx = segment_start_idx
        return kg

    def __len__(self):
        return self.nkpts

    def __iter__(self):
        for i in range(self.nkpts):
            yield self.kpoints_frac[i], self.weights[i]


@dataclass
class NscfResult:
    """Results from NSCF calculation."""
    eigenvalues: Dict[int, np.ndarray]  # k-point -> eigenvalues
    eigenvectors: Dict[int, np.ndarray]  # k-point -> wavefunctions
    converged: bool
    kgrid: KGrid


class Nscf:
    """
    Non-Self-Consistent Field (NSCF) calculator.

    Computes eigenvalues and eigenvectors on arbitrary k-points using
    a charge density pre-computed from an SCF calculation.
    """

    HA_TO_EV = 27.211386245988
    ANGSTROM_TO_BOHR = 1.8897259886

    def __init__(self, crystal: Crystal, kgrid: Union[KGrid, KPoints],
                 ecut: float, pseudopotentials: Dict[str, Pseudopotential],
                 xc_functional: str = 'LDA', scf_results: Optional[NscfResult] = None):
        """
        Initialize NSCF calculator.

        Args:
            crystal: Crystal structure
            kgrid: KGrid or KPoints object defining k-points
            ecut: Plane wave cutoff in Hartree
            pseudopotentials: Dictionary of pseudopotentials
            xc_functional: Exchange-correlation functional name
            scf_results: Optional pre-computed density from SCF
        """
        self.crystal = crystal

        # Handle ecut units (assume eV if > 10)
        if ecut > 10:
            self.ecut = ecut / self.HA_TO_EV
        else:
            self.ecut = ecut

        # Convert KPoints to KGrid if necessary
        if isinstance(kgrid, KPoints):
            # Convert KPoints object to KGrid
            kg = KGrid(kgrid.kpoints_frac, weights=kgrid.weights, crystal=crystal)
            self.kgrid = kg
        else:
            self.kgrid = kgrid

        if self.kgrid.crystal is None:
            self.kgrid.crystal = crystal

        self.pseudopotentials = pseudopotentials
        self.xc_functional = xc_functional
        self.scf_results = scf_results

        print(f"NSCF Calculator initialized:")
        print(f"  Cutoff energy: {self.ecut:.2f} Ha ({self.ecut * self.HA_TO_EV:.2f} eV)")
        print(f"  Number of k-points: {self.kgrid.nkpts}")
        print(f"  XC functional: {xc_functional}")

    def calculate(self, scf_hdf5_file: str = None,
                  verbose: bool = True) -> NscfResult:
        """
        Run NSCF calculation using converged density from SCF.

        Args:
            scf_hdf5_file: Path to SCF HDF5 file with converged density
            verbose: Print progress

        Returns:
            NscfResult with eigenvalues and eigenvectors
        """
        if verbose:
            print("\nStarting NSCF calculation...")

        # Load density from SCF HDF5 file if provided
        if scf_hdf5_file is not None:
            if verbose:
                print(f"Loading SCF density from {scf_hdf5_file}")
            scf_data = HDF5Output.read_scf_results(scf_hdf5_file)
            density = scf_data['density']
            fft_grid_shape = scf_data.get('fft_grid_shape')
        else:
            if verbose:
                print("Running SCF to get converged density...")
            # Run SCF on the standard k-grid to get density
            scf = SCFSolver(
                crystal=self.crystal,
                kpoints=self._to_kpoints(),
                ecut=self.ecut,
                pseudopotentials=self.pseudopotentials,
                xc_functional=self.xc_functional
            )

            scf_result = scf.solve(verbose=verbose)
            density = scf_result.density
            fft_grid_shape = tuple(scf.fft_grid.ng)

        # Create Hamiltonian for each NSCF k-point
        from .hamiltonian import Hamiltonian
        from .basis import PlaneWaveBasis, FFTGrid

        eigenvalues = {}
        eigenvectors = {}

        # Reshape density for effective potential calculation
        if fft_grid_shape is not None:
            rho_r = density.reshape(fft_grid_shape)
        else:
            rho_r = density

        # Create FFT grid for real-space operations
        fft_grid = FFTGrid(self.crystal, self.ecut)

        # Compute effective potential once (fixed from SCF)
        # For simplicity, we'll use the SCF Hamiltonian's potential calculation
        # This requires creating a temporary SCF solver or accessing its methods

        for ik, (k_frac, weight) in enumerate(self.kgrid):
            if verbose and ik % max(1, self.kgrid.nkpts // 10) == 0:
                print(f"  Computing eigenvalues at k-point {ik + 1}/{self.kgrid.nkpts}")

            # Get Cartesian k-point
            k_cart = k_frac @ self.crystal.reciprocal_cell

            # Create plane wave basis for this k-point
            pw_basis = PlaneWaveBasis(self.crystal, self.ecut, k_cart)

            # Create Hamiltonian
            h_kpt = Hamiltonian(
                crystal=self.crystal,
                pw_basis=pw_basis,
                fft_grid=fft_grid,
                pseudopotentials=self.pseudopotentials,
                xc_functional=self.xc_functional
            )

            # Build and diagonalize Hamiltonian with fixed density
            eigs, evecs = h_kpt.diagonalize(rho_r)

            eigenvalues[ik] = eigs
            eigenvectors[ik] = evecs

        result = NscfResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            converged=True,
            kgrid=self.kgrid
        )

        if verbose:
            print(f"NSCF calculation completed successfully")

        return result

    def _to_kpoints(self) -> KPoints:
        """Convert internal k-grid to KPoints object for SCF."""
        kpts = KPoints(self.crystal, (2, 2, 2), use_symmetry=False)
        kpts.kpoints_frac = self.kgrid.kpoints_frac
        kpts.kpoints_cart = self.kgrid.kpoints_cart
        kpts.weights = self.kgrid.weights
        kpts.nkpts = self.kgrid.nkpts
        return kpts

    def save_to_hdf5(self, filename: str, result: NscfResult) -> None:
        """
        Save NSCF results to HDF5 file.

        Args:
            filename: Output filename (e.g., 'nscf.h5')
            result: NscfResult object
        """
        output = HDF5Output(filename)
        output.write_nscf_results(
            crystal=self.crystal,
            kgrid=result.kgrid,
            eigenvalues=result.eigenvalues,
            eigenvectors=result.eigenvectors,
            converged=result.converged
        )


class DftElbands(Nscf):
    """
    DFT Band Structure Calculator.

    Derived from Nscf, computes band structure along a high-symmetry path.
    """

    def __init__(self, crystal: Crystal, special_points: Dict[str, np.ndarray],
                 path_str: str, npoints_per_segment: int = 50,
                 ecut: float = None, pseudopotentials: Dict[str, Pseudopotential] = None,
                 xc_functional: str = 'LDA'):
        """
        Initialize DFT band structure calculator.

        Args:
            crystal: Crystal structure
            special_points: Dict mapping point names to fractional coordinates
            path_str: Path specification like 'GXWLGK'
            npoints_per_segment: Number of k-points per segment
            ecut: Plane wave cutoff in Hartree
            pseudopotentials: Dictionary of pseudopotentials
            xc_functional: Exchange-correlation functional name
        """
        # Create k-grid along the path
        kgrid = KGrid.from_path(crystal, special_points, path_str, npoints_per_segment)

        # Initialize parent Nscf class
        super().__init__(
            crystal=crystal,
            kgrid=kgrid,
            ecut=ecut if ecut is not None else 10.0,
            pseudopotentials=pseudopotentials or {},
            xc_functional=xc_functional
        )

        self.special_points = special_points
        self.path_str = path_str
        self.npoints_per_segment = npoints_per_segment

        print(f"DFT Elbands Calculator initialized:")
        print(f"  Band structure path: {path_str}")
        print(f"  Total k-points along path: {self.kgrid.nkpts}")

    def save_to_hdf5(self, filename: str, result: NscfResult) -> None:
        """
        Save band structure results to HDF5 file.

        Args:
            filename: Output filename (e.g., 'dftelbands.h5')
            result: NscfResult object
        """
        output = HDF5Output(filename)
        output.write_dftelbands_results(
            crystal=self.crystal,
            kgrid=result.kgrid,
            eigenvalues=result.eigenvalues,
            eigenvectors=result.eigenvectors,
            path_str=self.path_str,
            special_points=self.special_points,
            converged=result.converged
        )

    def plot_bands(self, result: NscfResult, output_file: str = 'dftelbands.png',
                   fermi_energy: Optional[float] = None,
                   y_range: Optional[Tuple[float, float]] = None,
                   style: str = 'seabornv0_8-whitegrid',
                   n_bands_plot: Optional[int] = None) -> None:
        """
        Plot band structure.

        Args:
            result: NscfResult from calculate()
            output_file: Output PNG filename
            fermi_energy: Fermi energy to plot reference line (in Hartree)
            y_range: Y-axis range (emin, emax) in eV
            style: Matplotlib style name
            n_bands_plot: Number of bands to plot (default: all)
        """
        import matplotlib.pyplot as plt

        # Set style
        try:
            plt.style.use(style)
        except OSError:
            print(f"Warning: Style '{style}' not found, using default")

        # Convert eigenvalues to eV and collect data
        eigenvalues_eV = {}
        min_n_bands = float('inf')

        for ik in result.eigenvalues:
            eigenvalues_eV[ik] = result.eigenvalues[ik] * self.HA_TO_EV
            min_n_bands = min(min_n_bands, len(result.eigenvalues[ik]))

        # Determine number of bands to plot
        if n_bands_plot is None:
            n_bands_to_plot = min_n_bands
        else:
            n_bands_to_plot = min(n_bands_plot, min_n_bands)

        # Calculate distances along path
        distances = np.zeros(self.kgrid.nkpts)
        current_dist = 0.0

        # Find segment boundaries
        segment_indices = getattr(self.kgrid, 'segment_start_idx', None)
        if segment_indices is None:
            segment_indices = [0]

        for ik in range(1, self.kgrid.nkpts):
            # Distance between consecutive k-points
            dk = np.linalg.norm(
                self.kgrid.kpoints_frac[ik] - self.kgrid.kpoints_frac[ik - 1]
            )
            current_dist += dk
            distances[ik] = current_dist

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot bands
        for iband in range(n_bands_to_plot):
            band_energies = []
            for ik in range(self.kgrid.nkpts):
                if iband < len(eigenvalues_eV[ik]):
                    band_energies.append(eigenvalues_eV[ik][iband])
                else:
                    band_energies.append(np.nan)
            ax.plot(distances, band_energies, 'b-', linewidth=0.8, alpha=0.7)

        # Add Fermi energy reference line if provided
        if fermi_energy is not None:
            fermi_energy_eV = fermi_energy * self.HA_TO_EV
            ax.axhline(y=fermi_energy_eV, color='r', linestyle='--',
                      linewidth=1.0, label='Fermi level', alpha=0.7)

        # Add high-symmetry point labels and vertical lines
        labels = []
        for i, point_name in enumerate(self.path_str):
            if i == 0:
                labels.append((0.0, point_name))
            else:
                # Distance at segment boundary
                if i < len(segment_indices):
                    idx = segment_indices[i]
                    labels.append((distances[idx], point_name))

        # Remove duplicate labels
        seen_distances = set()
        unique_labels = []
        for dist, label in labels:
            if dist not in seen_distances:
                unique_labels.append((dist, label))
                seen_distances.add(dist)

        if unique_labels:
            label_distances, label_names = zip(*unique_labels)
            ax.set_xticks(label_distances)
            ax.set_xticklabels(label_names)

        # Add vertical lines at segment boundaries
        for idx in segment_indices[1:]:
            ax.axvline(x=distances[idx], color='k', linewidth=0.5, alpha=0.3)

        # Labels and formatting
        ax.set_xlabel('k-point path', fontsize=12)
        ax.set_ylabel('Energy (eV)', fontsize=12)
        ax.set_title(f'Band Structure: {self.path_str}', fontsize=14)
        ax.grid(True, alpha=0.3)

        if fermi_energy is not None:
            ax.legend()

        if y_range is not None:
            ax.set_ylim(y_range)

        # Save figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Band structure plot saved to {output_file}")
        plt.close()
