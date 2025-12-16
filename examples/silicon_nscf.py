#!/usr/bin/env python
"""
Non-Self-Consistent Field (NSCF) calculation for Silicon.

This example demonstrates computing eigenvalues and eigenvectors on a fine
k-point grid using the converged charge density from a prior SCF calculation.
"""

import os
import sys
import numpy as np
import h5py

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dft_pw import Crystal, KGrid, Nscf
from dft_pw.pseudopotential import read_upf


def main():
    """Run Silicon NSCF calculation."""
    print("=" * 60)
    print("NSCF Calculation for Silicon")
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

    # Create a fine k-point grid for NSCF
    nscf_kgrid = (4, 4, 4)  # Finer grid than SCF

    print(f"\nCalculation Parameters:")
    print(f"  Plane wave cutoff: {ecut} Ha ({ecut * 27.211:.1f} eV)")
    print(f"  NSCF k-point grid: {nscf_kgrid}")
    print(f"  SCF reference file: {scf_file}")

    # Create k-grid
    kgrid = KGrid.from_monkhorst_pack(crystal, nscf_kgrid)

    print(f"  Number of k-points: {kgrid.nkpts}")

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
        print("Skipping NSCF calculation for now")
        return None

    # Create NSCF calculator
    print("\nInitializing NSCF calculator...")
    nscf = Nscf(
        crystal=crystal,
        kgrid=kgrid,
        ecut=ecut,
        pseudopotentials=pseudopotentials,
        xc_functional='LDA'
    )

    # Run NSCF calculation
    print("\nStarting NSCF calculation...")
    result = nscf.calculate(scf_hdf5_file=scf_file, verbose=True)

    # Print eigenvalue information
    print("\n" + "=" * 60)
    print("NSCF Results")
    print("=" * 60)

    ha_to_ev = 27.211386245988
    n_bands = len(result.eigenvalues[0])

    print(f"\nEigenvalues at select k-points:")
    print("-" * 60)

    for ik in [0, len(kgrid) // 4, len(kgrid) // 2, 3 * len(kgrid) // 4]:
        if ik < len(kgrid):
            k_frac = kgrid.kpoints_frac[ik]
            eigs = result.eigenvalues[ik]

            print(f"\nk-point {ik}: ({k_frac[0]:.3f}, {k_frac[1]:.3f}, {k_frac[2]:.3f})")
            print("  Band    Energy (Ha)      Energy (eV)")
            for iband in range(min(n_bands, 10)):  # Show first 10 bands
                e_ha = eigs[iband]
                e_ev = e_ha * ha_to_ev
                print(f"  {iband + 1:3d}    {e_ha:12.6f}    {e_ev:12.6f}")

    # Save to HDF5
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nscf.h5')
    print("\n" + "=" * 60)
    print("Saving Results to HDF5")
    print("=" * 60)
    nscf.save_to_hdf5(output_file, result)

    # Print HDF5 contents
    print("\n" + "=" * 60)
    print("HDF5 Output File Contents")
    print("=" * 60)
    print(f"\nFile: {output_file}")
    print_hdf5_contents(output_file)

    return result


def print_hdf5_contents(filename):
    """Print summary of HDF5 file contents."""
    with h5py.File(filename, 'r') as f:
        print(f"\nFile attributes:")
        for key, val in f.attrs.items():
            print(f"  {key}: {val}")

        print(f"\nGroups and datasets:")

        def print_structure(name, obj):
            indent = "  " * (name.count('/') + 1)
            if isinstance(obj, h5py.Group):
                print(f"{indent}[Group] {name}/")
                for key, val in obj.attrs.items():
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

        print(f"\nNumber of k-points: {f['kpoints'].attrs['n_kpoints']}")

        # Eigenvalue info
        eig_keys = list(f['eigenvalues'].keys())
        if eig_keys:
            sample_eigs = f[f'eigenvalues/{eig_keys[0]}'][:]
            print(f"Number of bands: {len(sample_eigs)}")

        # Wavefunction info
        wfc_keys = list(f['wavefunctions'].keys())
        if wfc_keys:
            sample_wfc = f[f'wavefunctions/{wfc_keys[0]}']
            print(f"Wavefunction shape per k-point: {sample_wfc.shape}")
            print(f"  (n_planewaves, n_bands)")


if __name__ == '__main__':
    result = main()
