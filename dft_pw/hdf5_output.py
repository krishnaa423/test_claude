"""
HDF5 output module for DFT calculations.

Saves wavefunctions, charge density, and energies to HDF5 format.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from typing import Dict, Optional
from datetime import datetime


class HDF5Output:
    """
    HDF5 output handler for DFT calculations.
    """

    def __init__(self, filename: str = "scf.h5"):
        """
        Initialize HDF5 output handler.

        Args:
            filename: Output filename (default: scf.h5)
        """
        self.filename = filename

    def write_scf_results(self,
                          crystal,
                          kpoints,
                          eigenvalues: Dict[int, np.ndarray],
                          eigenvectors: Dict[int, np.ndarray],
                          occupations: Dict[int, np.ndarray],
                          density: np.ndarray,
                          total_energy: float,
                          fermi_energy: float,
                          band_gap: Optional[float] = None,
                          converged: bool = True,
                          n_iterations: int = 0,
                          fft_grid_shape: tuple = None):
        """
        Write SCF results to HDF5 file.

        Args:
            crystal: Crystal structure
            kpoints: KPoints object
            eigenvalues: Dict mapping k-point index to eigenvalues
            eigenvectors: Dict mapping k-point index to eigenvectors (wavefunctions)
            occupations: Dict mapping k-point index to occupations
            density: Charge density on real-space grid
            total_energy: Total energy in Hartree
            fermi_energy: Fermi energy in Hartree
            band_gap: Band gap in eV (optional)
            converged: Whether SCF converged
            n_iterations: Number of SCF iterations
            fft_grid_shape: Shape of FFT grid (ng1, ng2, ng3)
        """
        with h5py.File(self.filename, 'w') as f:
            # Metadata
            f.attrs['title'] = 'DFT Plane Wave Calculation'
            f.attrs['created'] = datetime.now().isoformat()
            f.attrs['converged'] = converged
            f.attrs['n_iterations'] = n_iterations

            # Energies group
            energies = f.create_group('energies')
            energies.attrs['total_energy_Ha'] = total_energy
            energies.attrs['total_energy_eV'] = total_energy * 27.211386245988
            energies.attrs['fermi_energy_Ha'] = fermi_energy
            energies.attrs['fermi_energy_eV'] = fermi_energy * 27.211386245988
            if band_gap is not None:
                energies.attrs['band_gap_eV'] = band_gap

            # Crystal structure group
            crystal_grp = f.create_group('crystal')
            crystal_grp.create_dataset('cell', data=crystal.cell)
            crystal_grp.attrs['volume_bohr3'] = crystal.volume
            crystal_grp.attrs['n_atoms'] = crystal.num_atoms

            # Atomic positions
            positions = np.array([atom.position for atom in crystal.atoms])
            symbols = [atom.symbol for atom in crystal.atoms]
            crystal_grp.create_dataset('positions_fractional', data=positions)
            crystal_grp.create_dataset('positions_cartesian', data=crystal.get_cartesian_positions())
            # Store symbols as fixed-length strings
            dt = h5py.string_dtype(encoding='utf-8')
            crystal_grp.create_dataset('symbols', data=symbols, dtype=dt)

            # K-points group
            kpts_grp = f.create_group('kpoints')
            kpts_grp.attrs['n_kpoints'] = kpoints.nkpts
            kpts_grp.attrs['grid'] = kpoints.grid
            kpts_grp.create_dataset('kpoints_fractional', data=kpoints.kpoints_frac)
            kpts_grp.create_dataset('kpoints_cartesian', data=kpoints.kpoints_cart)
            kpts_grp.create_dataset('weights', data=kpoints.weights)

            # Eigenvalues group
            eigen_grp = f.create_group('eigenvalues')
            for ik in eigenvalues:
                eigen_grp.create_dataset(f'kpoint_{ik}', data=eigenvalues[ik])
                eigen_grp[f'kpoint_{ik}'].attrs['units'] = 'Hartree'

            # Occupations group
            occ_grp = f.create_group('occupations')
            for ik in occupations:
                occ_grp.create_dataset(f'kpoint_{ik}', data=occupations[ik])

            # Wavefunctions group
            wfc_grp = f.create_group('wavefunctions')
            wfc_grp.attrs['description'] = 'Plane wave coefficients for each k-point and band'
            for ik in eigenvectors:
                # eigenvectors[ik] has shape (npw, nbands)
                wfc_grp.create_dataset(f'kpoint_{ik}', data=eigenvectors[ik])
                wfc_grp[f'kpoint_{ik}'].attrs['shape'] = '(n_planewaves, n_bands)'

            # Charge density group
            density_grp = f.create_group('density')
            if fft_grid_shape is not None:
                density_3d = density.reshape(fft_grid_shape)
                density_grp.create_dataset('rho', data=density_3d)
                density_grp.attrs['grid_shape'] = fft_grid_shape
            else:
                density_grp.create_dataset('rho', data=density)
            density_grp.attrs['units'] = 'electrons/bohr^3'
            density_grp.attrs['total_charge'] = np.sum(density) * crystal.volume / len(density)

        print(f"Results written to {self.filename}")

    @staticmethod
    def read_scf_results(filename: str) -> dict:
        """
        Read SCF results from HDF5 file.

        Args:
            filename: Input filename

        Returns:
            Dictionary with all stored data
        """
        results = {}

        with h5py.File(filename, 'r') as f:
            # Metadata
            results['converged'] = f.attrs['converged']
            results['n_iterations'] = f.attrs['n_iterations']

            # Energies
            results['total_energy_Ha'] = f['energies'].attrs['total_energy_Ha']
            results['total_energy_eV'] = f['energies'].attrs['total_energy_eV']
            results['fermi_energy_Ha'] = f['energies'].attrs['fermi_energy_Ha']
            results['fermi_energy_eV'] = f['energies'].attrs['fermi_energy_eV']
            if 'band_gap_eV' in f['energies'].attrs:
                results['band_gap_eV'] = f['energies'].attrs['band_gap_eV']

            # Crystal
            results['cell'] = f['crystal/cell'][:]
            results['positions'] = f['crystal/positions_fractional'][:]
            results['symbols'] = [s.decode() if isinstance(s, bytes) else s
                                  for s in f['crystal/symbols'][:]]

            # K-points
            results['kpoints_frac'] = f['kpoints/kpoints_fractional'][:]
            results['kpoints_cart'] = f['kpoints/kpoints_cartesian'][:]
            results['weights'] = f['kpoints/weights'][:]

            # Eigenvalues
            results['eigenvalues'] = {}
            for key in f['eigenvalues'].keys():
                ik = int(key.split('_')[1])
                results['eigenvalues'][ik] = f[f'eigenvalues/{key}'][:]

            # Occupations
            results['occupations'] = {}
            for key in f['occupations'].keys():
                ik = int(key.split('_')[1])
                results['occupations'][ik] = f[f'occupations/{key}'][:]

            # Wavefunctions
            results['wavefunctions'] = {}
            for key in f['wavefunctions'].keys():
                ik = int(key.split('_')[1])
                results['wavefunctions'][ik] = f[f'wavefunctions/{key}'][:]

            # Density
            results['density'] = f['density/rho'][:]

        return results


def save_scf_to_hdf5(filename: str, scf_result, crystal, kpoints,
                     eigenvectors: Dict[int, np.ndarray], fft_grid) -> None:
    """
    Convenience function to save SCF results to HDF5.

    Args:
        filename: Output filename
        scf_result: SCFResult object
        crystal: Crystal object
        kpoints: KPoints object
        eigenvectors: Wavefunctions from SCF
        fft_grid: FFTGrid object
    """
    output = HDF5Output(filename)
    output.write_scf_results(
        crystal=crystal,
        kpoints=kpoints,
        eigenvalues=scf_result.eigenvalues,
        eigenvectors=eigenvectors,
        occupations=scf_result.occupations,
        density=scf_result.density,
        total_energy=scf_result.total_energy,
        fermi_energy=scf_result.fermi_energy,
        band_gap=scf_result.band_gap if hasattr(scf_result, 'band_gap') else None,
        converged=scf_result.converged,
        n_iterations=scf_result.n_iterations,
        fft_grid_shape=tuple(fft_grid.ng)
    )


def plot_charge_density(filename: str, output_png: str = "charge_density.png",
                        slice_axis: int = 2, slice_index: int = None) -> None:
    """
    Create a pcolormesh plot of charge density from HDF5 file.

    Args:
        filename: Input HDF5 filename
        output_png: Output PNG filename (default: charge_density.png)
        slice_axis: Axis to slice along (0, 1, or 2) - default 2 (z-axis)
        slice_index: Index along slice_axis (default: middle of grid)
    """
    with h5py.File(filename, 'r') as f:
        density = f['density/rho'][:]
        grid_shape = density.shape

    # Get slice index (default to middle)
    if slice_index is None:
        slice_index = grid_shape[slice_axis] // 2

    # Extract 2D slice
    if slice_axis == 0:
        density_2d = density[slice_index, :, :]
        xlabel, ylabel = 'y', 'z'
    elif slice_axis == 1:
        density_2d = density[:, slice_index, :]
        xlabel, ylabel = 'x', 'z'
    else:  # slice_axis == 2
        density_2d = density[:, :, slice_index]
        xlabel, ylabel = 'x', 'y'

    # Create pcolormesh plot with viridis colormap
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(density_2d.T, cmap='viridis', shading='auto')
    cbar = fig.colorbar(mesh, ax=ax, label='Charge density (electrons/bohr³)')

    ax.set_xlabel(f'{xlabel} (grid points)')
    ax.set_ylabel(f'{ylabel} (grid points)')
    ax.set_title(f'Charge Density (slice at {["x", "y", "z"][slice_axis]}={slice_index})')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()

    print(f"Charge density plot saved to {output_png}")
