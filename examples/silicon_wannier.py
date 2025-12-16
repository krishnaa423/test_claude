#!/usr/bin/env python
"""
Wannier function calculation for Silicon.

This example demonstrates computing maximally localized Wannier functions
using the converged wavefunctions from an NSCF calculation.
"""

import os
import sys
import numpy as np
import h5py

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dft_pw import Crystal, KGrid, Nscf, Wannier
from dft_pw.wannier import save_wannier_to_hdf5, read_wannier_from_hdf5
from dft_pw.pseudopotential import read_upf


def main():
    """Run Silicon Wannier function calculation."""
    print("=" * 60)
    print("Wannier Function Calculation for Silicon")
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
    ecut = 10.0  # Hartree - must match SCF/NSCF calculation
    scf_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scf.h5')

    # Create a k-point grid for Wannier functions
    # Use a smaller grid for faster computation
    wannier_kgrid = (2, 2, 2)

    print(f"\nCalculation Parameters:")
    print(f"  Plane wave cutoff: {ecut} Ha ({ecut * 27.211:.1f} eV)")
    print(f"  Wannier k-point grid: {wannier_kgrid}")
    print(f"  SCF reference file: {scf_file}")

    # Create k-grid
    kgrid = KGrid.from_monkhorst_pack(crystal, wannier_kgrid)
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
            continue

        print(f"  Loading PP for {symbol}: {pp_path}")
        pseudopotentials[symbol] = read_upf(pp_path)

    # Check if SCF file exists
    if not os.path.exists(scf_file):
        print(f"\nWarning: SCF file not found: {scf_file}")
        print("Please run 'python silicon.py' first to generate the SCF results")
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
    print("Running NSCF calculation...")
    nscf_result = nscf.calculate(scf_hdf5_file=scf_file, verbose=True)

    # Print NSCF results
    print("\n" + "=" * 60)
    print("NSCF Results")
    print("=" * 60)
    ha_to_ev = 27.211386245988
    print(f"\nNumber of k-points: {nscf.kgrid.nkpts}")
    print(f"Number of bands: {len(nscf_result.eigenvalues[0])}")

    # Create Wannier function calculator
    print("\n" + "=" * 60)
    print("Wannier Function Calculation")
    print("=" * 60)

    n_wann = 4  # Number of Wannier functions (valence bands for Si)

    print(f"\nInitializing Wannier calculator...")
    print(f"  Number of Wannier functions: {n_wann}")

    w = Wannier(nscf_result, n_wann=n_wann, crystal=crystal, kgrid=kgrid)

    # Compute Wannier functions
    print("\nComputing maximally localized Wannier functions...")
    wannier_result = w.compute(max_iter=30, conv_tol=1e-5, verbose=True)

    # Print Wannier results
    print("\n" + "=" * 60)
    print("Wannier Function Results")
    print("=" * 60)

    print(f"\nConverged: {wannier_result.converged}")
    print(f"Total spread: {wannier_result.total_spread:.8f} Bohr²")
    print(f"Number of Wannier functions: {wannier_result.n_wann}")

    # Compute Wannier centers
    print("\nComputing Wannier function centers...")
    wannier_centers = w.compute_wannier_centers(wannier_result.U_matrix)

    # Compute physical spreads
    total_spread, spreads = w.compute_spreads_physical(wannier_centers)
    print(f"Physical spread: {total_spread:.8f}")
    print(f"Average spread per Wannier function: {total_spread / n_wann:.8f}")

    # Print Wannier centers
    print("\nWannier function centers (fractional coordinates):")
    print("-" * 60)

    for ik in range(min(4, nscf.kgrid.nkpts)):  # Show first 4 k-points
        k_frac = nscf.kgrid.kpoints_frac[ik]
        centers = wannier_centers[ik]

        print(f"\nk-point {ik}: ({k_frac[0]:.3f}, {k_frac[1]:.3f}, {k_frac[2]:.3f})")
        print("  Wannier Center (fractional)")
        for iw in range(n_wann):
            c = centers[iw]
            print(f"    W{iw+1}: ({c[0]:8.4f}, {c[1]:8.4f}, {c[2]:8.4f})")

    # Save to HDF5
    output_h5 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wannier.h5')
    print("\n" + "=" * 60)
    print("Saving Results to HDF5")
    print("=" * 60)

    save_wannier_to_hdf5(output_h5, wannier_result, wannier_centers=wannier_centers,
                        crystal=crystal, kgrid=kgrid)

    # Print HDF5 contents
    print("\n" + "=" * 60)
    print("HDF5 Output File Contents")
    print("=" * 60)
    print(f"\nFile: {output_h5}")
    print_hdf5_contents(output_h5)

    # Analyze U matrices
    print("\n" + "=" * 60)
    print("U Matrix Analysis")
    print("=" * 60)

    print(f"\nU matrices for each k-point:")
    for ik in range(min(2, nscf.kgrid.nkpts)):
        U = wannier_result.U_matrix[ik]
        print(f"\nk-point {ik}:")
        print(f"  Shape: {U.shape}")

        # Check unitarity
        UU = np.conj(U.T) @ U
        deviation = np.linalg.norm(UU - np.eye(n_wann))
        print(f"  Unitarity deviation: {deviation:.2e}")

        # Print first few elements
        print(f"  |U| elements (first row):")
        print(f"    {np.abs(U[0, :])}")

    return wannier_result


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

        print(f"\nTotal spread: {f.attrs['total_spread']:.8f}")
        print(f"Number of Wannier functions: {f.attrs['n_wann']}")
        print(f"Number of bands: {f.attrs['n_bands']}")
        print(f"Converged: {f.attrs['converged']}")

        # U matrix info
        u_grp = f['U_matrices']
        u_keys = list(u_grp.keys())
        if u_keys:
            sample_u = u_grp[u_keys[0]][:]
            print(f"\nU matrix shape per k-point: {sample_u.shape}")

        # Wannier centers info
        if 'wannier_centers' in f:
            centers_grp = f['wannier_centers']
            centers_keys = list(centers_grp.keys())
            if centers_keys:
                sample_centers = centers_grp[centers_keys[0]][:]
                print(f"Wannier centers shape per k-point: {sample_centers.shape}")


if __name__ == '__main__':
    result = main()
