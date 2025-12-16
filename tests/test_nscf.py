"""
Test cases for NSCF and DftElbands classes.
"""

import pytest
import numpy as np
import os
import tempfile
from dft_pw import Crystal, KGrid, Nscf, DftElbands
from dft_pw.pseudopotential import read_upf


@pytest.fixture
def simple_crystal():
    """Create a simple cubic crystal for testing."""
    a = 10.0  # Bohr
    cell = a * np.eye(3)
    from dft_pw import Atom
    atoms = [Atom('H', [0.0, 0.0, 0.0])]
    return Crystal(cell, atoms, units='bohr')


@pytest.fixture
def fcc_crystal():
    """Create FCC crystal structure."""
    return Crystal.diamond(5.43, 'Si', units='angstrom')


class TestKGrid:
    """Test KGrid class."""

    def test_kgrid_creation_explicit(self, simple_crystal):
        """Test explicit k-grid creation."""
        kpoints_frac = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
        ])
        kgrid = KGrid(kpoints_frac, crystal=simple_crystal)

        assert kgrid.nkpts == 3
        assert np.allclose(kgrid.kpoints_frac, kpoints_frac)
        assert np.allclose(kgrid.weights, [1/3, 1/3, 1/3])

    def test_kgrid_custom_weights(self, simple_crystal):
        """Test k-grid with custom weights."""
        kpoints_frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        weights = np.array([0.7, 0.3])

        kgrid = KGrid(kpoints_frac, weights=weights, crystal=simple_crystal)

        assert kgrid.nkpts == 2
        assert np.allclose(kgrid.weights, weights)
        assert np.allclose(np.sum(kgrid.weights), 1.0)

    def test_kgrid_monkhorst_pack(self, fcc_crystal):
        """Test Monkhorst-Pack grid generation."""
        kgrid = KGrid.from_monkhorst_pack(fcc_crystal, grid=(2, 2, 2))

        assert kgrid.nkpts == 8
        assert np.allclose(np.sum(kgrid.weights), 1.0)

    def test_kgrid_special_points_path(self, fcc_crystal):
        """Test k-path through special points."""
        special_points = {
            'G': np.array([0.0, 0.0, 0.0]),
            'X': np.array([0.5, 0.0, 0.5]),
        }

        kgrid = KGrid.from_path(fcc_crystal, special_points, 'GX', npoints_per_segment=5)

        # Should have 6 points: start at G, then 5 interpolated points
        assert kgrid.nkpts == 6
        # Check that first point is at Gamma
        assert np.allclose(kgrid.kpoints_frac[0], [0.0, 0.0, 0.0])
        # Check that last point is at X
        assert np.allclose(kgrid.kpoints_frac[-1], [0.5, 0.0, 0.5])

    def test_kgrid_cartesian_conversion(self, fcc_crystal):
        """Test conversion to Cartesian coordinates."""
        kgrid = KGrid.from_monkhorst_pack(fcc_crystal, grid=(2, 2, 2))

        k_cart = kgrid.kpoints_cart

        # Check shape
        assert k_cart.shape == (8, 3)

        # Check that Cartesian is related to fractional by reciprocal lattice
        for i in range(kgrid.nkpts):
            k_cart_computed = kgrid.kpoints_frac[i] @ fcc_crystal.reciprocal_cell
            assert np.allclose(k_cart[i], k_cart_computed)


class TestNscf:
    """Test NSCF class."""

    def test_nscf_initialization(self, simple_crystal):
        """Test NSCF initialization."""
        kgrid = KGrid.from_monkhorst_pack(simple_crystal, grid=(2, 2, 2))

        nscf = Nscf(
            crystal=simple_crystal,
            kgrid=kgrid,
            ecut=5.0,
            pseudopotentials={},
            xc_functional='LDA'
        )

        assert nscf.crystal is simple_crystal
        assert nscf.kgrid is kgrid
        assert nscf.ecut == pytest.approx(5.0)
        assert nscf.xc_functional == 'LDA'

    def test_nscf_ecut_conversion(self, simple_crystal):
        """Test ecut unit conversion (eV to Hartree)."""
        kgrid = KGrid.from_monkhorst_pack(simple_crystal, grid=(2, 2, 2))

        # Test with eV (large value)
        nscf = Nscf(
            crystal=simple_crystal,
            kgrid=kgrid,
            ecut=100.0,  # Interpreted as eV
            pseudopotentials={},
            xc_functional='LDA'
        )

        # Should be converted to Hartree
        assert nscf.ecut < 10  # Much smaller in Hartree
        assert nscf.ecut == pytest.approx(100.0 / 27.211386245988)

    def test_nscf_hdf5_save(self, simple_crystal, tmp_path):
        """Test saving NSCF results to HDF5."""
        import h5py

        kgrid = KGrid.from_monkhorst_pack(simple_crystal, grid=(2, 2, 2))

        # Create mock results
        eigenvalues = {i: np.array([1.0, 2.0, 3.0]) for i in range(kgrid.nkpts)}
        eigenvectors = {i: np.random.randn(5, 3) + 1j * np.random.randn(5, 3)
                        for i in range(kgrid.nkpts)}

        from dft_pw import NscfResult
        result = NscfResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            converged=True,
            kgrid=kgrid
        )

        nscf = Nscf(
            crystal=simple_crystal,
            kgrid=kgrid,
            ecut=5.0,
            pseudopotentials={},
            xc_functional='LDA'
        )

        output_file = os.path.join(str(tmp_path), 'nscf.h5')
        nscf.save_to_hdf5(output_file, result)

        # Verify file exists and has expected structure
        assert os.path.exists(output_file)

        with h5py.File(output_file, 'r') as f:
            assert 'eigenvalues' in f
            assert 'wavefunctions' in f
            assert 'crystal' in f
            assert 'kpoints' in f
            assert f.attrs['type'] == 'NSCF'


class TestDftElbands:
    """Test DftElbands class."""

    def test_dft_elbands_initialization(self, fcc_crystal):
        """Test DftElbands initialization."""
        special_points = {
            'G': np.array([0.0, 0.0, 0.0]),
            'X': np.array([0.5, 0.0, 0.5]),
            'W': np.array([0.5, 0.25, 0.75]),
            'L': np.array([0.5, 0.5, 0.5]),
        }

        elbands = DftElbands(
            crystal=fcc_crystal,
            special_points=special_points,
            path_str='GXWLG',
            npoints_per_segment=10,
            ecut=5.0,
            pseudopotentials={},
            xc_functional='LDA'
        )

        assert elbands.crystal is fcc_crystal
        assert elbands.path_str == 'GXWLG'
        assert elbands.npoints_per_segment == 10
        # Should have 4 segments with 10 points each = 40 points + 1 (first G)
        # Actually: G->X: 11 points (G + 10 interpolated), X->W: 10 interp, W->L: 10 interp, L->G: 10 interp
        # But the implementation removes duplicates at segment boundaries
        assert elbands.kgrid.nkpts > 0

    def test_dft_elbands_path_generation(self, fcc_crystal):
        """Test band structure path generation."""
        special_points = {
            'G': np.array([0.0, 0.0, 0.0]),
            'X': np.array([0.5, 0.0, 0.5]),
        }

        elbands = DftElbands(
            crystal=fcc_crystal,
            special_points=special_points,
            path_str='GX',
            npoints_per_segment=5,
            ecut=5.0,
            pseudopotentials={},
            xc_functional='LDA'
        )

        # Check k-path
        kpoints = elbands.kgrid.kpoints_frac
        assert kpoints[0].sum() == pytest.approx(0.0)  # First point is Gamma
        assert np.allclose(kpoints[-1], [0.5, 0.0, 0.5])  # Last point is X

    def test_dft_elbands_hdf5_save(self, fcc_crystal, tmp_path):
        """Test saving DftElbands results to HDF5."""
        import h5py

        special_points = {
            'G': np.array([0.0, 0.0, 0.0]),
            'X': np.array([0.5, 0.0, 0.5]),
        }

        elbands = DftElbands(
            crystal=fcc_crystal,
            special_points=special_points,
            path_str='GX',
            npoints_per_segment=5,
            ecut=5.0,
            pseudopotentials={},
            xc_functional='LDA'
        )

        # Create mock results
        n_kpts = elbands.kgrid.nkpts
        eigenvalues = {i: np.array([1.0, 2.0, 3.0]) for i in range(n_kpts)}
        eigenvectors = {i: np.random.randn(5, 3) + 1j * np.random.randn(5, 3)
                        for i in range(n_kpts)}

        from dft_pw import NscfResult
        result = NscfResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            converged=True,
            kgrid=elbands.kgrid
        )

        output_file = os.path.join(str(tmp_path), 'dftelbands.h5')
        elbands.save_to_hdf5(output_file, result)

        # Verify file exists and has expected structure
        assert os.path.exists(output_file)

        with h5py.File(output_file, 'r') as f:
            assert 'eigenvalues' in f
            assert 'wavefunctions' in f
            assert 'crystal' in f
            assert 'kpoints' in f
            assert 'special_points' in f
            assert f.attrs['type'] == 'DFT_ELBANDS'
            assert f.attrs['path'] == 'GX'

    def test_dft_elbands_plot(self, fcc_crystal, tmp_path):
        """Test band structure plotting."""
        special_points = {
            'G': np.array([0.0, 0.0, 0.0]),
            'X': np.array([0.5, 0.0, 0.5]),
        }

        elbands = DftElbands(
            crystal=fcc_crystal,
            special_points=special_points,
            path_str='GX',
            npoints_per_segment=5,
            ecut=5.0,
            pseudopotentials={},
            xc_functional='LDA'
        )

        # Create mock results
        n_kpts = elbands.kgrid.nkpts
        eigenvalues = {i: np.array([1.0, 2.0, 3.0]) for i in range(n_kpts)}
        eigenvectors = {i: np.random.randn(5, 3) + 1j * np.random.randn(5, 3)
                        for i in range(n_kpts)}

        from dft_pw import NscfResult
        result = NscfResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            converged=True,
            kgrid=elbands.kgrid
        )

        output_file = os.path.join(str(tmp_path), 'dftelbands.png')
        elbands.plot_bands(result, output_file=output_file, fermi_energy=-0.5)

        # Verify plot file was created
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
