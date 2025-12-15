"""
Tests for k-point generation module.
"""

import numpy as np
import pytest
from dft_pw.crystal import Crystal, Atom
from dft_pw.kpoints import KPoints


class TestKPoints:
    """Tests for KPoints class."""

    @pytest.fixture
    def silicon_crystal(self):
        """Create silicon diamond structure."""
        return Crystal.diamond(5.43, 'Si', units='angstrom')

    def test_full_grid_creation(self, silicon_crystal):
        """Test full k-point grid without symmetry."""
        kpts = KPoints(silicon_crystal, (2, 2, 2), use_symmetry=False)

        # 2x2x2 grid should have 8 k-points
        assert kpts.nkpts == 8

    def test_weights_sum_to_one(self, silicon_crystal):
        """Test that k-point weights sum to 1."""
        kpts = KPoints(silicon_crystal, (4, 4, 4), use_symmetry=False)

        total_weight = np.sum(kpts.weights)
        assert total_weight == pytest.approx(1.0)

    def test_symmetry_reduction(self, silicon_crystal):
        """Test k-point reduction with symmetry."""
        kpts_full = KPoints(silicon_crystal, (4, 4, 4), use_symmetry=False)
        kpts_sym = KPoints(silicon_crystal, (4, 4, 4), use_symmetry=True)

        # Symmetry should reduce the number of k-points
        # (if spglib is available)
        assert kpts_sym.nkpts <= kpts_full.nkpts

        # Weights should still sum to 1
        assert np.sum(kpts_sym.weights) == pytest.approx(1.0)

    def test_gamma_centered_grid(self, silicon_crystal):
        """Test that unshifted grid includes Gamma point."""
        kpts = KPoints(silicon_crystal, (3, 3, 3), shift=(0, 0, 0), use_symmetry=False)

        # Check if Gamma point (0,0,0) is in the grid
        has_gamma = False
        for k_frac in kpts.kpoints_frac:
            if np.allclose(k_frac, [0, 0, 0], atol=1e-10):
                has_gamma = True
                break

        assert has_gamma, "Gamma point should be in unshifted grid"

    def test_cartesian_conversion(self, silicon_crystal):
        """Test conversion to Cartesian coordinates."""
        kpts = KPoints(silicon_crystal, (2, 2, 2), use_symmetry=False)

        # Cartesian k-points should match fractional * reciprocal_cell
        for i in range(kpts.nkpts):
            k_cart_expected = kpts.kpoints_frac[i] @ silicon_crystal.reciprocal_cell
            np.testing.assert_array_almost_equal(
                kpts.kpoints_cart[i], k_cart_expected
            )

    def test_iteration(self, silicon_crystal):
        """Test iterating over k-points."""
        kpts = KPoints(silicon_crystal, (2, 2, 2), use_symmetry=False)

        count = 0
        for k_cart, weight in kpts:
            assert k_cart.shape == (3,)
            assert weight > 0
            count += 1

        assert count == kpts.nkpts

    def test_len(self, silicon_crystal):
        """Test len() on KPoints."""
        kpts = KPoints(silicon_crystal, (3, 3, 3), use_symmetry=False)
        assert len(kpts) == kpts.nkpts

    def test_high_symmetry_points_available(self, silicon_crystal):
        """Test that high symmetry points are defined."""
        kpts = KPoints(silicon_crystal, (2, 2, 2))

        special = kpts._get_cubic_special_points()

        assert 'G' in special  # Gamma
        assert 'X' in special
        assert 'L' in special
        assert 'K' in special

        # Gamma should be at origin
        np.testing.assert_array_almost_equal(special['G'], [0, 0, 0])
