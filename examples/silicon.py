#!/usr/bin/env python
"""
Silicon DFT calculation example.

This example demonstrates a basic DFT calculation for bulk Silicon
using the plane wave code, with HDF5 output.
"""

import os
import sys
import numpy as np
import h5py

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dft_pw import Crystal, DFTCalculator
from dft_pw.hdf5_output import plot_charge_density


def main():
    """Run Silicon DFT calculation."""
    print("=" * 60)
    print("DFT Plane Wave Calculation for Silicon")
    print("=" * 60)

    # Silicon lattice constant (Angstrom)
    a_si = 5.43

    # Create silicon diamond structure
    crystal = Crystal.diamond(a_si, 'Si', units='angstrom')

    print(f"\nCrystal Structure:")
    print(f"  Material: Silicon (diamond structure)")
    print(f"  Lattice constant: {a_si} Angstrom")
    print(f"  Number of atoms: {crystal.num_atoms}")
    print(f"  Cell volume: {crystal.volume / Crystal.ANGSTROM_TO_BOHR**3:.2f} Angstrom^3")

    # Pseudopotential directory
    pp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'pseudopotentials')

    # Calculation parameters
    ecut = 10.0  # Hartree (~272 eV) - low for testing, use 20-30 Ha for production
    kgrid = (2, 2, 2)  # Small k-grid for testing

    print(f"\nCalculation Parameters:")
    print(f"  Plane wave cutoff: {ecut} Ha ({ecut * 27.211:.1f} eV)")
    print(f"  K-point grid: {kgrid}")
    print(f"  Exchange-correlation: LDA")

    # Create calculator
    calc = DFTCalculator(
        crystal=crystal,
        ecut=ecut,
        kgrid=kgrid,
        xc='LDA',
        pseudopotential_dir=pp_dir,
        smearing=0.01,  # Hartree
        use_symmetry=True
    )

    # Output HDF5 file path
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scf.h5')

    # Run calculation with HDF5 output
    print("\nStarting SCF calculation...")
    result = calc.calculate(
        max_scf_iter=50,
        scf_tol=1e-5,
        mixing='pulay',
        mixing_alpha=0.3,
        verbose=True,
        save_hdf5=output_file
    )

    # Print eigenvalues at each k-point
    print("\nEigenvalues at each k-point:")
    print("-" * 40)
    ha_to_ev = 27.211386245988

    for ik in range(len(calc.kpoints)):
        k_frac = calc.kpoints.kpoints_frac[ik]
        weight = calc.kpoints.weights[ik]
        eigs = result.eigenvalues[ik]

        print(f"\nk-point {ik + 1}: ({k_frac[0]:.3f}, {k_frac[1]:.3f}, {k_frac[2]:.3f}), "
              f"weight = {weight:.4f}")
        print("  Band   Occupation   Energy (Ha)   Energy (eV)")
        for iband, (e, occ) in enumerate(zip(eigs, result.occupations[ik])):
            print(f"  {iband + 1:4d}   {occ:8.4f}     {e:12.6f}   {e * ha_to_ev:12.6f}")

    # Show HDF5 file contents
    print("\n" + "=" * 60)
    print("HDF5 Output File Contents")
    print("=" * 60)
    print(f"\nFile: {output_file}")
    print_hdf5_contents(output_file)

    # Generate charge density plot
    print("\n" + "=" * 60)
    print("Generating Charge Density Plot")
    print("=" * 60)
    png_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'charge_density.png')
    plot_charge_density(output_file, png_file)

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

        print(f"\nTotal Energy: {f['energies'].attrs['total_energy_Ha']:.8f} Ha")
        print(f"Fermi Energy: {f['energies'].attrs['fermi_energy_Ha']:.8f} Ha")
        if 'band_gap_eV' in f['energies'].attrs:
            print(f"Band Gap: {f['energies'].attrs['band_gap_eV']:.4f} eV")

        print(f"\nCharge density grid shape: {f['density/rho'].shape}")
        print(f"Total charge: {f['density'].attrs['total_charge']:.4f} electrons")

        print(f"\nNumber of k-points: {f['kpoints'].attrs['n_kpoints']}")
        print(f"K-grid: {f['kpoints'].attrs['grid']}")

        # Wavefunction info
        wfc_keys = list(f['wavefunctions'].keys())
        if wfc_keys:
            sample_wfc = f[f'wavefunctions/{wfc_keys[0]}']
            print(f"\nWavefunction shape per k-point: {sample_wfc.shape}")
            print(f"  (n_planewaves, n_bands)")


if __name__ == '__main__':
    result = main()
