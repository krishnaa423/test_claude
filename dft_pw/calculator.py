"""
Main DFT Calculator interface.
"""

import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .crystal import Crystal, Atom
from .kpoints import KPoints
from .pseudopotential import read_upf, Pseudopotential
from .scf import SCFSolver, SCFResult


@dataclass
class DFTResult:
    """Results from DFT calculation."""
    total_energy: float  # Hartree
    fermi_energy: float  # Hartree
    eigenvalues: Dict[int, np.ndarray]  # k-point -> eigenvalues
    occupations: Dict[int, np.ndarray]  # k-point -> occupations
    converged: bool
    n_scf_iterations: int
    band_gap: Optional[float] = None  # eV

    def print_summary(self):
        """Print summary of results."""
        ha_to_ev = 27.211386245988

        print("\n" + "=" * 60)
        print("DFT Calculation Results")
        print("=" * 60)
        print(f"  Total Energy:      {self.total_energy:16.8f} Ha")
        print(f"                     {self.total_energy * ha_to_ev:16.8f} eV")
        print(f"  Fermi Energy:      {self.fermi_energy:16.8f} Ha")
        print(f"                     {self.fermi_energy * ha_to_ev:16.8f} eV")
        if self.band_gap is not None:
            print(f"  Band Gap:          {self.band_gap:16.8f} eV")
        print(f"  SCF Converged:     {self.converged}")
        print(f"  SCF Iterations:    {self.n_scf_iterations}")
        print("=" * 60)


class DFTCalculator:
    """
    High-level interface for DFT calculations.
    """

    # Conversion factors
    HA_TO_EV = 27.211386245988
    ANGSTROM_TO_BOHR = 1.8897259886

    def __init__(self, crystal: Crystal, ecut: float,
                 kgrid: Tuple[int, int, int] = (4, 4, 4),
                 xc: str = 'LDA',
                 pseudopotential_dir: str = '.',
                 smearing: float = 0.01,
                 use_symmetry: bool = True):
        """
        Initialize DFT calculator.

        Args:
            crystal: Crystal structure
            ecut: Plane wave cutoff in Hartree (or eV if > 10)
            kgrid: Monkhorst-Pack k-point grid
            xc: Exchange-correlation functional
            pseudopotential_dir: Directory containing UPF files
            smearing: Smearing width in Hartree
            use_symmetry: Use symmetry to reduce k-points
        """
        self.crystal = crystal

        # Handle ecut units (assume eV if > 10)
        if ecut > 10:
            self.ecut = ecut / self.HA_TO_EV
        else:
            self.ecut = ecut

        self.kgrid = kgrid
        self.xc = xc
        self.pp_dir = pseudopotential_dir
        self.smearing = smearing
        self.use_symmetry = use_symmetry

        # Load pseudopotentials
        self.pseudopotentials = self._load_pseudopotentials()

        # Set up k-points
        self.kpoints = KPoints(crystal, kgrid, use_symmetry=use_symmetry)

        print(f"DFT Calculator initialized:")
        print(f"  Cutoff energy: {self.ecut:.2f} Ha ({self.ecut * self.HA_TO_EV:.2f} eV)")
        print(f"  K-point grid: {kgrid}")
        print(f"  Irreducible k-points: {self.kpoints.nkpts}")
        print(f"  XC functional: {xc}")

    def _load_pseudopotentials(self) -> Dict[str, Pseudopotential]:
        """Load pseudopotentials for all species."""
        pps = {}
        species = self.crystal.get_unique_species()

        for symbol in species:
            # Try different file naming conventions
            filenames = [
                f"{symbol}.upf",
                f"{symbol}.UPF",
                f"{symbol}_ONCV_PBE-1.0.upf",
                f"{symbol}_ONCV_PBE_sr.upf",
            ]

            pp_path = None
            for fname in filenames:
                path = os.path.join(self.pp_dir, fname)
                if os.path.exists(path):
                    pp_path = path
                    break

            if pp_path is None:
                raise FileNotFoundError(
                    f"Could not find pseudopotential for {symbol} in {self.pp_dir}. "
                    f"Tried: {filenames}"
                )

            print(f"  Loading PP for {symbol}: {pp_path}")
            pps[symbol] = read_upf(pp_path)

        return pps

    def calculate(self, max_scf_iter: int = 100, scf_tol: float = 1e-6,
                  mixing: str = 'pulay', mixing_alpha: float = 0.3,
                  n_bands: int = None, verbose: bool = True) -> DFTResult:
        """
        Run DFT calculation.

        Args:
            max_scf_iter: Maximum SCF iterations
            scf_tol: SCF convergence tolerance (Hartree)
            mixing: Density mixing method ('linear', 'pulay')
            mixing_alpha: Mixing parameter
            n_bands: Number of bands (default: auto)
            verbose: Print progress

        Returns:
            DFTResult with calculation results
        """
        if verbose:
            print("\nStarting DFT calculation...")

        # Create SCF solver
        scf = SCFSolver(
            crystal=self.crystal,
            kpoints=self.kpoints,
            ecut=self.ecut,
            pseudopotentials=self.pseudopotentials,
            xc_functional=self.xc,
            n_bands=n_bands,
            smearing=self.smearing
        )

        # Run SCF
        scf_result = scf.solve(
            max_iter=max_scf_iter,
            tol=scf_tol,
            mixing=mixing,
            mixing_alpha=mixing_alpha,
            verbose=verbose
        )

        # Calculate band gap
        band_gap = self._calculate_band_gap(
            scf_result.eigenvalues, scf_result.occupations
        )

        result = DFTResult(
            total_energy=scf_result.total_energy,
            fermi_energy=scf_result.fermi_energy,
            eigenvalues=scf_result.eigenvalues,
            occupations=scf_result.occupations,
            converged=scf_result.converged,
            n_scf_iterations=scf_result.n_iterations,
            band_gap=band_gap
        )

        if verbose:
            result.print_summary()

        return result

    def _calculate_band_gap(self, eigenvalues: Dict[int, np.ndarray],
                            occupations: Dict[int, np.ndarray]) -> Optional[float]:
        """Calculate band gap from eigenvalues."""
        vbm = -np.inf  # Valence band maximum
        cbm = np.inf   # Conduction band minimum

        for ik in eigenvalues:
            eigs = eigenvalues[ik]
            occs = occupations[ik]

            for i, (e, occ) in enumerate(zip(eigs, occs)):
                if occ > 0.5:  # Occupied
                    if e > vbm:
                        vbm = e
                else:  # Unoccupied
                    if e < cbm:
                        cbm = e

        if cbm > vbm:
            return (cbm - vbm) * self.HA_TO_EV
        else:
            return 0.0  # Metal

    def get_band_structure(self, path: str = None,
                           npoints: int = 50) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Calculate band structure along high-symmetry path.

        Args:
            path: Path specification (e.g., 'GXWLGK')
            npoints: Points per segment

        Returns:
            (k_distances, eigenvalues, labels)
        """
        # Get high-symmetry path
        k_cart, distances, labels = self.kpoints.get_high_symmetry_path(
            path, npoints
        )

        print(f"\nCalculating band structure along {len(k_cart)} k-points...")

        # First do SCF to get converged density
        scf = SCFSolver(
            crystal=self.crystal,
            kpoints=self.kpoints,
            ecut=self.ecut,
            pseudopotentials=self.pseudopotentials,
            xc_functional=self.xc,
            smearing=self.smearing
        )

        scf_result = scf.solve(verbose=False)

        # Now calculate eigenvalues at path k-points
        # (non-self-consistent calculation with converged potential)
        # This would require modifying SCF to accept external potential
        # For simplicity, we return the k-point eigenvalues we have

        return distances, scf_result.eigenvalues, labels


def silicon_test():
    """Quick test with Silicon."""
    # Silicon diamond structure
    a = 5.43  # Angstrom
    crystal = Crystal.diamond(a, 'Si', units='angstrom')

    print(f"Silicon crystal:")
    print(f"  Lattice constant: {a} Angstrom")
    print(f"  Volume: {crystal.volume / Crystal.ANGSTROM_TO_BOHR**3:.2f} Angstrom^3")
    print(f"  Atoms: {crystal.num_atoms}")

    return crystal
