"""
Tests for plane wave basis module.
"""

import numpy as np
import pytest
from dft_pw.crystal import Crystal, Atom
from dft_pw.basis import PlaneWaveBasis, FFTGrid


class TestPlaneWaveBasis:
    """Tests for PlaneWaveBasis class."""

    @pytest.fixture
    def simple_crystal(self):
        """Create a simple cubic crystal for testing."""
        a = 10.0  # Bohr
        cell = a * np.eye(3)
        atoms = [Atom('Si', [0.0, 0.0, 0.0])]
        return Crystal(cell, atoms, units='bohr')

    def test_basis_creation(self, simple_crystal):
        """Test basis set creation."""
        ecut = 5.0  # Hartree
        pw = PlaneWaveBasis(simple_crystal, ecut)

        assert pw.npw > 0
        assert len(pw.g_vectors) == pw.npw
        assert len(pw.kinetic_energies) == pw.npw

    def test_kinetic_energies_within_cutoff(self, simple_crystal):
        """Test that all kinetic energies are within cutoff."""
        ecut = 5.0
        pw = PlaneWaveBasis(simple_crystal, ecut)

        assert np.all(pw.kinetic_energies <= ecut)

    def test_kinetic_energies_sorted(self, simple_crystal):
        """Test that kinetic energies are sorted."""
        ecut = 5.0
        pw = PlaneWaveBasis(simple_crystal, ecut)

        assert np.all(pw.kinetic_energies[:-1] <= pw.kinetic_energies[1:])

    def test_gamma_point_includes_zero(self, simple_crystal):
        """Test that Gamma point basis includes G=0."""
        ecut = 5.0
        pw = PlaneWaveBasis(simple_crystal, ecut, k=np.zeros(3))

        # G=0 should have zero kinetic energy
        assert pw.kinetic_energies[0] == pytest.approx(0.0, abs=1e-10)

    def test_kpoint_shift(self, simple_crystal):
        """Test basis with non-zero k-point."""
        ecut = 5.0
        k = np.array([0.1, 0.0, 0.0])
        pw = PlaneWaveBasis(simple_crystal, ecut, k=k)

        # With k != 0, minimum kinetic energy should be non-zero
        # (unless k happens to align with a G vector)
        assert pw.npw > 0

    def test_higher_cutoff_more_pw(self, simple_crystal):
        """Test that higher cutoff gives more plane waves."""
        pw_low = PlaneWaveBasis(simple_crystal, ecut=2.0)
        pw_high = PlaneWaveBasis(simple_crystal, ecut=10.0)

        assert pw_high.npw > pw_low.npw

    def test_kinetic_diagonal(self, simple_crystal):
        """Test kinetic energy diagonal retrieval."""
        ecut = 5.0
        pw = PlaneWaveBasis(simple_crystal, ecut)

        diag = pw.get_kinetic_diagonal()
        np.testing.assert_array_equal(diag, pw.kinetic_energies)


class TestFFTGrid:
    """Tests for FFTGrid class."""

    @pytest.fixture
    def simple_crystal(self):
        """Create a simple cubic crystal for testing."""
        a = 10.0  # Bohr
        cell = a * np.eye(3)
        atoms = [Atom('Si', [0.0, 0.0, 0.0])]
        return Crystal(cell, atoms, units='bohr')

    def test_grid_creation(self, simple_crystal):
        """Test FFT grid creation."""
        ecut = 5.0
        fft = FFTGrid(simple_crystal, ecut)

        assert fft.ng.shape == (3,)
        assert fft.nr.shape == (3,)
        assert fft.nrtot == np.prod(fft.nr)

    def test_fft_roundtrip(self, simple_crystal):
        """Test FFT forward-inverse roundtrip."""
        ecut = 5.0
        fft = FFTGrid(simple_crystal, ecut)

        # Create test function in real space
        f_r = np.random.randn(fft.nrtot) + 1j * np.random.randn(fft.nrtot)

        # Forward then inverse
        f_g = fft.to_reciprocal_space(f_r)
        f_r_back = fft.to_real_space(f_g)

        np.testing.assert_array_almost_equal(f_r, f_r_back)

    def test_pw_grid_mapping(self, simple_crystal):
        """Test mapping between PW basis and FFT grid."""
        ecut = 3.0
        fft = FFTGrid(simple_crystal, ecut)
        pw = PlaneWaveBasis(simple_crystal, ecut)

        # Create test coefficients
        coeffs = np.random.randn(pw.npw) + 1j * np.random.randn(pw.npw)

        # Map to grid and back
        f_g = fft.pw_to_grid(coeffs, pw)
        coeffs_back = fft.grid_to_pw(f_g, pw)

        np.testing.assert_array_almost_equal(coeffs, coeffs_back)
