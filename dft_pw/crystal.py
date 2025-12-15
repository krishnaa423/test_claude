"""
Crystal structure module for DFT calculations.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Atom:
    """Represents an atom in the crystal structure."""
    symbol: str
    position: np.ndarray  # Fractional coordinates
    atomic_number: int = None

    def __post_init__(self):
        self.position = np.array(self.position, dtype=np.float64)
        if self.atomic_number is None:
            self.atomic_number = ATOMIC_NUMBERS.get(self.symbol, 0)


# Atomic numbers for common elements
ATOMIC_NUMBERS = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
    'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
    'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
    'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
}


class Crystal:
    """
    Represents a crystal structure with unit cell and atoms.

    Attributes:
        cell: 3x3 matrix with lattice vectors as rows (in Bohr)
        atoms: List of Atom objects
    """

    # Conversion factor: Angstrom to Bohr
    ANGSTROM_TO_BOHR = 1.8897259886

    def __init__(self, cell: np.ndarray, atoms: List[Atom], units: str = 'angstrom'):
        """
        Initialize crystal structure.

        Args:
            cell: 3x3 matrix with lattice vectors as rows
            atoms: List of Atom objects with fractional coordinates
            units: 'angstrom' or 'bohr' for cell vectors
        """
        self.cell = np.array(cell, dtype=np.float64)
        if units.lower() == 'angstrom':
            self.cell *= self.ANGSTROM_TO_BOHR
        self.atoms = atoms

        # Compute reciprocal lattice vectors
        self._compute_reciprocal_cell()

    def _compute_reciprocal_cell(self):
        """Compute reciprocal lattice vectors (2*pi factor included)."""
        self.volume = np.abs(np.linalg.det(self.cell))
        self.reciprocal_cell = 2 * np.pi * np.linalg.inv(self.cell).T

    @property
    def num_atoms(self) -> int:
        """Number of atoms in the unit cell."""
        return len(self.atoms)

    @property
    def num_electrons(self) -> int:
        """Total number of valence electrons (assumes neutral atoms)."""
        # This will be updated by pseudopotentials
        return sum(atom.atomic_number for atom in self.atoms)

    def get_cartesian_positions(self) -> np.ndarray:
        """Get atomic positions in Cartesian coordinates (Bohr)."""
        positions = np.array([atom.position for atom in self.atoms])
        return positions @ self.cell

    def get_fractional_positions(self) -> np.ndarray:
        """Get atomic positions in fractional coordinates."""
        return np.array([atom.position for atom in self.atoms])

    def get_species(self) -> List[str]:
        """Get list of atomic species symbols."""
        return [atom.symbol for atom in self.atoms]

    def get_unique_species(self) -> List[str]:
        """Get list of unique atomic species."""
        seen = set()
        unique = []
        for atom in self.atoms:
            if atom.symbol not in seen:
                seen.add(atom.symbol)
                unique.append(atom.symbol)
        return unique

    def get_spglib_cell(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get cell in spglib format.

        Returns:
            (lattice, positions, numbers) tuple for spglib
        """
        lattice = self.cell / self.ANGSTROM_TO_BOHR  # Convert back to Angstrom for spglib
        positions = self.get_fractional_positions()
        numbers = np.array([atom.atomic_number for atom in self.atoms])
        return (lattice, positions, numbers)

    def get_structure_factor_positions(self, G: np.ndarray) -> np.ndarray:
        """
        Compute structure factor phase for G-vectors.

        Args:
            G: G-vectors in Cartesian coordinates (Bohr^-1)

        Returns:
            Complex structure factor S(G) = sum_atoms exp(-i G . tau)
        """
        tau = self.get_cartesian_positions()
        # G has shape (nG, 3), tau has shape (natoms, 3)
        # Result is (nG, natoms)
        phases = np.exp(-1j * G @ tau.T)
        return phases

    @classmethod
    def from_parameters(cls, a: float, b: float, c: float,
                        alpha: float, beta: float, gamma: float,
                        atoms: List[Atom], units: str = 'angstrom'):
        """
        Create crystal from lattice parameters.

        Args:
            a, b, c: Lattice vector lengths
            alpha, beta, gamma: Angles in degrees
            atoms: List of Atom objects
            units: 'angstrom' or 'bohr'
        """
        alpha_r = np.radians(alpha)
        beta_r = np.radians(beta)
        gamma_r = np.radians(gamma)

        # Build cell matrix
        cos_alpha = np.cos(alpha_r)
        cos_beta = np.cos(beta_r)
        cos_gamma = np.cos(gamma_r)
        sin_gamma = np.sin(gamma_r)

        cell = np.zeros((3, 3))
        cell[0, 0] = a
        cell[1, 0] = b * cos_gamma
        cell[1, 1] = b * sin_gamma
        cell[2, 0] = c * cos_beta
        cell[2, 1] = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        cell[2, 2] = c * np.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2
                                  + 2 * cos_alpha * cos_beta * cos_gamma) / sin_gamma

        return cls(cell, atoms, units)

    @classmethod
    def diamond(cls, a: float, symbol: str = 'Si', units: str = 'angstrom'):
        """
        Create a diamond structure crystal (like Si or C).

        Args:
            a: Lattice constant
            symbol: Atomic symbol
            units: 'angstrom' or 'bohr'
        """
        cell = a * np.array([
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5]
        ])

        atoms = [
            Atom(symbol, [0.00, 0.00, 0.00]),
            Atom(symbol, [0.25, 0.25, 0.25]),
        ]

        return cls(cell, atoms, units)

    @classmethod
    def fcc(cls, a: float, symbol: str, units: str = 'angstrom'):
        """Create an FCC structure."""
        cell = a * np.array([
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5]
        ])
        atoms = [Atom(symbol, [0.0, 0.0, 0.0])]
        return cls(cell, atoms, units)

    @classmethod
    def bcc(cls, a: float, symbol: str, units: str = 'angstrom'):
        """Create a BCC structure."""
        cell = a * np.array([
            [-0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, -0.5]
        ])
        atoms = [Atom(symbol, [0.0, 0.0, 0.0])]
        return cls(cell, atoms, units)
