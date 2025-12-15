#!/usr/bin/env python
"""
Silicon DFT calculation example.

This example demonstrates a basic DFT calculation for bulk Silicon
using the plane wave code.
"""

import os
import sys
import numpy as np

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dft_pw import Crystal, DFTCalculator


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

    # Run calculation
    print("\nStarting SCF calculation...")
    result = calc.calculate(
        max_scf_iter=50,
        scf_tol=1e-5,
        mixing='pulay',
        mixing_alpha=0.3,
        verbose=True
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

    return result


if __name__ == '__main__':
    result = main()
