#!/usr/bin/env python
"""
Band structure calculation for Silicon using high-symmetry path.

This example demonstrates calculating and plotting band structure along
a high-symmetry path in the Brillouin zone.
"""

import os
import sys
import numpy as np
import h5py

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dft_pw import Crystal, DftElbands
from dft_pw.pseudopotential import read_upf


def main():
    """Run Silicon band structure calculation."""
    print("=" * 60)
    print("Band Structure Calculation for Silicon")
    print("=" * 60)

    # Silicon lattice constant (Angstrom)
    a_si = 5.43

    # Create silicon diamond structure
    crystal = Crystal.diamond(a_si, 'Si', units='angstrom')

    print(f"\nCrystal Structure:")
    print(f"  Material: Silicon (diamond structure)")
    print(f"  Lattice constant: {a_si} Angstrom")
    print(f"  Number of atoms: {crystal.num_atoms}")

    # Pseudopotential directory
    pp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'pseudopotentials')

    # Calculation parameters
    ecut = 10.0  # Hartree - must match SCF calculation
    scf_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scf.h5')

    print(f"\nCalculation Parameters:")
    print(f"  Plane wave cutoff: {ecut} Ha ({ecut * 27.211:.1f} eV)")
    print(f"  Band structure path: GXWLG")
    print(f"  Points per segment: 50")
    print(f"  SCF reference file: {scf_file}")

    # Define high-symmetry points for FCC/Diamond structure
    special_points = {
        'G': np.array([0.0, 0.0, 0.0]),       # Gamma
        'X': np.array([0.5, 0.0, 0.5]),       # X
        'W': np.array([0.5, 0.25, 0.75]),     # W
        'L': np.array([0.5, 0.5, 0.5]),       # L
        'K': np.array([0.375, 0.375, 0.75]),  # K
    }

    # Load pseudopotentials
    print("\nLoading pseudopotentials...")
    pseudopotentials = {}
    species = crystal.get_unique_species()

    for symbol in species:
        filenames = [
            f"{symbol}.upf",
            f"{symbol}.UPF",
            f"{symbol}_ONCV_PBE-1.0.upf",
            f"{symbol}_ONCV_PBE_sr.upf",
        ]

        pp_path = None
        for fname in filenames:
            path = os.path.join(pp_dir, fname)
            if os.path.exists(path):
                pp_path = path
                break

        if pp_path is None:
            print(f"Warning: Could not find pseudopotential for {symbol}")
            print(f"  Tried: {filenames}")
            continue

        print(f"  Loading PP for {symbol}: {pp_path}")
        pseudopotentials[symbol] = read_upf(pp_path)

    # Check if SCF file exists
    if not os.path.exists(scf_file):
        print(f"\nWarning: SCF file not found: {scf_file}")
        print("Please run 'python silicon.py' first to generate the SCF results")
        print("Skipping band structure calculation for now")
        return None

    # Create DFT band structure calculator
    print("\nInitializing band structure calculator...")
    elbands = DftElbands(
        crystal=crystal,
        special_points=special_points,
        path_str='GXWLG',
        npoints_per_segment=50,
        ecut=ecut,
        pseudopotentials=pseudopotentials,
        xc_functional='LDA'
    )

    # Run band structure calculation
    print("\nStarting band structure calculation...")
    result = elbands.calculate(scf_hdf5_file=scf_file, verbose=True)

    # Print band structure information
    print("\n" + "=" * 60)
    print("Band Structure Results")
    print("=" * 60)

    ha_to_ev = 27.211386245988
    n_bands = len(result.eigenvalues[0])

    print(f"\nBand structure path: GXWLG")
    print(f"Total k-points along path: {elbands.kgrid.nkpts}")
    print(f"Number of bands: {n_bands}")

    # Get Fermi energy from SCF file if available
    fermi_energy = None
    try:
        with h5py.File(scf_file, 'r') as f:
            if 'energies' in f and 'fermi_energy_Ha' in f['energies'].attrs:
                fermi_energy = f['energies'].attrs['fermi_energy_Ha']
                print(f"Fermi energy: {fermi_energy:.6f} Ha ({fermi_energy * ha_to_ev:.6f} eV)")
    except:
        pass

    # Print sample eigenvalues
    print(f"\nSample eigenvalues at selected k-points:")
    print("-" * 60)

    for ik in [0, elbands.kgrid.nkpts // 4, elbands.kgrid.nkpts // 2]:
        k_frac = elbands.kgrid.kpoints_frac[ik]
        eigs = result.eigenvalues[ik]

        print(f"\nk-point {ik}: ({k_frac[0]:.3f}, {k_frac[1]:.3f}, {k_frac[2]:.3f})")
        print("  Band    Energy (Ha)      Energy (eV)")
        for iband in range(min(n_bands, 8)):  # Show first 8 bands
            e_ha = eigs[iband]
            e_ev = e_ha * ha_to_ev
            print(f"  {iband + 1:3d}    {e_ha:12.6f}    {e_ev:12.6f}")

    # Save to HDF5
    output_h5 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dftelbands.h5')
    print("\n" + "=" * 60)
    print("Saving Results to HDF5")
    print("=" * 60)
    elbands.save_to_hdf5(output_h5, result)

    # Create band structure plot
    output_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dftelbands.png')
    print("\n" + "=" * 60)
    print("Generating Band Structure Plot")
    print("=" * 60)
    elbands.plot_bands(result, output_file=output_png, fermi_energy=fermi_energy)

    # Print HDF5 contents
    print("\n" + "=" * 60)
    print("HDF5 Output File Contents")
    print("=" * 60)
    print(f"\nFile: {output_h5}")
    print_hdf5_contents(output_h5)

    return result


def print_hdf5_contents(filename):
    """Print summary of HDF5 file contents."""
    with h5py.File(filename, 'r') as f:
        print(f"\nFile attributes:")
        for key, val in f.attrs.items():
            if isinstance(val, bytes):
                val = val.decode()
            print(f"  {key}: {val}")

        print(f"\nGroups and datasets:")

        def print_structure(name, obj):
            indent = "  " * (name.count('/') + 1)
            if isinstance(obj, h5py.Group):
                print(f"{indent}[Group] {name}/")
                for key, val in obj.attrs.items():
                    if isinstance(val, bytes):
                        val = val.decode()
                    print(f"{indent}  @{key}: {val}")
            elif isinstance(obj, h5py.Dataset):
                shape_str = str(obj.shape)
                dtype_str = str(obj.dtype)
                print(f"{indent}[Dataset] {name}: shape={shape_str}, dtype={dtype_str}")

        f.visititems(print_structure)

        # Print some sample data
        print("\n" + "-" * 40)
        print("Sample Data:")
        print("-" * 40)

        print(f"\nBand structure path: {f.attrs['path'].decode() if isinstance(f.attrs['path'], bytes) else f.attrs['path']}")
        print(f"Number of k-points: {f['kpoints'].attrs['n_kpoints']}")

        # Eigenvalue info
        eig_keys = list(f['eigenvalues'].keys())
        if eig_keys:
            sample_eigs = f[f'eigenvalues/{eig_keys[0]}'][:]
            print(f"Number of bands: {len(sample_eigs)}")

        # Special points
        if 'special_points' in f:
            print(f"Special points in path: {list(f['special_points'].keys())}")


if __name__ == '__main__':
    result = main()
