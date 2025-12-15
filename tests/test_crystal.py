"""
Tests for crystal structure module.
"""

import numpy as np
import pytest
from dft_pw.crystal import Crystal, Atom, ATOMIC_NUMBERS


class TestAtom:
    """Tests for Atom class."""

    def test_atom_creation(self):
        """Test basic atom creation."""
        atom = Atom('Si', [0.0, 0.0, 0.0])
        assert atom.symbol == 'Si'
        assert atom.atomic_number == 14
        np.testing.assert_array_equal(atom.position, [0.0, 0.0, 0.0])

    def test_atom_fractional_coords(self):
        """Test atom with fractional coordinates."""
        atom = Atom('C', [0.25, 0.25, 0.25])
        np.testing.assert_array_almost_equal(atom.position, [0.25, 0.25, 0.25])


class TestCrystal:
    """Tests for Crystal class."""

    def test_cubic_cell(self):
        """Test simple cubic cell."""
        a = 5.0  # Angstrom
        cell = a * np.eye(3)
        atoms = [Atom('Si', [0.0, 0.0, 0.0])]
        crystal = Crystal(cell, atoms, units='angstrom')

        # Check cell is converted to Bohr
        expected_cell = a * Crystal.ANGSTROM_TO_BOHR * np.eye(3)
        np.testing.assert_array_almost_equal(crystal.cell, expected_cell)

    def test_diamond_structure(self):
        """Test diamond structure creation."""
        a = 5.43  # Silicon lattice constant
        crystal = Crystal.diamond(a, 'Si', units='angstrom')

        assert crystal.num_atoms == 2
        assert crystal.get_species() == ['Si', 'Si']

    def test_reciprocal_lattice(self):
        """Test reciprocal lattice calculation."""
        a = 5.0
        cell = a * np.eye(3)
        atoms = [Atom('Si', [0.0, 0.0, 0.0])]
        crystal = Crystal(cell, atoms, units='bohr')

        # For cubic cell, reciprocal vectors should be 2*pi/a
        expected = (2 * np.pi / a) * np.eye(3)
        np.testing.assert_array_almost_equal(crystal.reciprocal_cell, expected)

    def test_volume(self):
        """Test volume calculation."""
        a = 10.0  # Bohr
        cell = a * np.eye(3)
        atoms = [Atom('Si', [0.0, 0.0, 0.0])]
        crystal = Crystal(cell, atoms, units='bohr')

        assert crystal.volume == pytest.approx(a**3)

    def test_cartesian_positions(self):
        """Test conversion from fractional to Cartesian coordinates."""
        a = 10.0
        cell = a * np.eye(3)
        atoms = [
            Atom('Si', [0.0, 0.0, 0.0]),
            Atom('Si', [0.5, 0.5, 0.5]),
        ]
        crystal = Crystal(cell, atoms, units='bohr')

        cart = crystal.get_cartesian_positions()
        expected = np.array([
            [0.0, 0.0, 0.0],
            [5.0, 5.0, 5.0]
        ])
        np.testing.assert_array_almost_equal(cart, expected)

    def test_spglib_cell_format(self):
        """Test spglib cell format output."""
        a = 5.43
        crystal = Crystal.diamond(a, 'Si', units='angstrom')

        lattice, positions, numbers = crystal.get_spglib_cell()

        # Lattice should be in Angstrom for spglib
        assert lattice.shape == (3, 3)
        assert positions.shape == (2, 3)
        assert len(numbers) == 2
        assert all(n == 14 for n in numbers)  # Silicon atomic number

    def test_fcc_structure(self):
        """Test FCC structure creation."""
        a = 4.0
        crystal = Crystal.fcc(a, 'Al', units='angstrom')

        assert crystal.num_atoms == 1

    def test_bcc_structure(self):
        """Test BCC structure creation."""
        a = 3.0
        crystal = Crystal.bcc(a, 'Fe', units='angstrom')

        assert crystal.num_atoms == 1


class TestAtomicNumbers:
    """Tests for atomic number lookup."""

    def test_common_elements(self):
        """Test atomic numbers for common elements."""
        assert ATOMIC_NUMBERS['H'] == 1
        assert ATOMIC_NUMBERS['C'] == 6
        assert ATOMIC_NUMBERS['Si'] == 14
        assert ATOMIC_NUMBERS['Fe'] == 26
