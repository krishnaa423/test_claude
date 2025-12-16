"""
Test cases for Wannier function calculation.
"""

import pytest
import numpy as np
import os
import tempfile
import h5py

from dft_pw import Crystal, KGrid, Wannier, WannierResult, NscfResult
from dft_pw.wannier import save_wannier_to_hdf5, read_wannier_from_hdf5


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


@pytest.fixture
def mock_nscf_result(simple_crystal):
    """Create a mock NSCF result for testing."""
    n_kpts = 8
    n_bands = 10
    n_pw = 50

    kgrid = KGrid.from_monkhorst_pack(simple_crystal, grid=(2, 2, 2))

    eigenvalues = {}
    eigenvectors = {}

    for ik in range(n_kpts):
        # Mock eigenvalues
        eigs = np.linspace(-1.0, 1.0, n_bands)
        eigenvalues[ik] = eigs

        # Mock eigenvectors (plane wave coefficients)
        # Should be (n_planewaves, n_bands) with variable n_planewaves
        n_pw_k = n_pw + np.random.randint(-5, 5)
        evecs = np.random.randn(n_pw_k, n_bands) + 1j * np.random.randn(n_pw_k, n_bands)
        # Normalize
        for ib in range(n_bands):
            evecs[:, ib] /= np.linalg.norm(evecs[:, ib])
        eigenvectors[ik] = evecs

    return NscfResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        converged=True,
        kgrid=kgrid
    )


class TestWannierInitialization:
    """Test Wannier initialization."""

    def test_wannier_creation(self, mock_nscf_result, simple_crystal):
        """Test Wannier function initialization."""
        w = Wannier(mock_nscf_result, n_wann=4, crystal=simple_crystal)

        assert w.n_bands == 10
        assert w.n_wann == 4
        assert w.n_kpts == 8

    def test_wannier_invalid_n_wann(self, mock_nscf_result):
        """Test that n_wann cannot exceed n_bands."""
        with pytest.raises(ValueError):
            Wannier(mock_nscf_result, n_wann=20)  # More than n_bands=10

    def test_wannier_attributes(self, mock_nscf_result, simple_crystal):
        """Test Wannier attributes."""
        w = Wannier(mock_nscf_result, n_wann=4, crystal=simple_crystal)

        assert w.crystal is simple_crystal
        assert w.n_bands == len(mock_nscf_result.eigenvalues[0])
        assert w.n_wann == 4
        assert w.n_kpts == len(mock_nscf_result.eigenvalues)


class TestWannierComputation:
    """Test Wannier function computation."""

    def test_overlap_matrix_computation(self, mock_nscf_result, simple_crystal):
        """Test overlap matrix computation."""
        w = Wannier(mock_nscf_result, n_wann=4, crystal=simple_crystal)

        M_matrix = w._compute_overlap_matrix()

        # Check that overlap matrices exist for all k-point pairs
        assert len(M_matrix) > 0

        # Check shapes
        for (ik, ik_next), M in M_matrix.items():
            assert M.shape == (w.n_bands, w.n_bands)
            # Overlap should be approximately unitary-like (diagonal close to 1)
            assert np.all(np.abs(np.diag(M)) <= 1.1)

    def test_u_matrix_initialization(self, mock_nscf_result, simple_crystal):
        """Test U matrix initialization."""
        w = Wannier(mock_nscf_result, n_wann=4, crystal=simple_crystal)

        U_matrix = w._initialize_u_matrix()

        # Check U matrix for each k-point
        assert len(U_matrix) == w.n_kpts

        for ik, U in U_matrix.items():
            # Should be (n_bands, n_wann)
            assert U.shape == (w.n_bands, w.n_wann)
            # Should be approximately unitary (orthogonal columns)
            UU = np.conj(U.T) @ U
            assert np.allclose(UU, np.eye(w.n_wann), atol=1e-6)

    def test_compute_spreads(self, mock_nscf_result, simple_crystal):
        """Test spread computation."""
        w = Wannier(mock_nscf_result, n_wann=4, crystal=simple_crystal)

        U_matrix = w._initialize_u_matrix()
        spreads = w._compute_spreads(U_matrix)

        # Check spreads for each k-point
        assert len(spreads) == w.n_kpts

        for ik, omega in spreads.items():
            # Should have n_wann spreads
            assert len(omega) == w.n_wann
            # Spreads should be positive
            assert np.all(omega >= 0)

    def test_gradient_step(self, mock_nscf_result, simple_crystal):
        """Test gradient descent step."""
        w = Wannier(mock_nscf_result, n_wann=4, crystal=simple_crystal)

        U_matrix = w._initialize_u_matrix()
        M_matrix = w._compute_overlap_matrix()

        U_new = w._gradient_step(U_matrix, M_matrix, learning_rate=0.1)

        # Check that U matrices are updated
        assert len(U_new) == w.n_kpts

        for ik, U in U_new.items():
            # Should still be (n_bands, n_wann)
            assert U.shape == (w.n_bands, w.n_wann)
            # Should still be unitary (orthogonal)
            UU = np.conj(U.T) @ U
            assert np.allclose(UU, np.eye(w.n_wann), atol=1e-5)

    def test_full_computation(self, mock_nscf_result, simple_crystal):
        """Test full Wannier computation."""
        w = Wannier(mock_nscf_result, n_wann=4, crystal=simple_crystal)

        result = w.compute(max_iter=10, conv_tol=1e-4, verbose=False)

        # Check result
        assert isinstance(result, WannierResult)
        assert result.converged or len(result.U_matrix) > 0
        assert result.n_bands == 10
        assert result.n_wann == 4
        assert result.total_spread > 0

        # Check U matrices
        assert len(result.U_matrix) == w.n_kpts
        for ik, U in result.U_matrix.items():
            assert U.shape == (w.n_bands, w.n_wann)

    def test_wannier_centers_computation(self, mock_nscf_result, simple_crystal):
        """Test Wannier centers computation."""
        w = Wannier(mock_nscf_result, n_wann=4, crystal=simple_crystal)

        result = w.compute(max_iter=5, verbose=False)
        centers = w.compute_wannier_centers(result.U_matrix)

        # Check centers
        assert len(centers) == w.n_kpts

        for ik, center_array in centers.items():
            # Should be (n_wann, 3)
            assert center_array.shape == (w.n_wann, 3)
            # Centers should be in reasonable range (fractional coordinates)
            assert np.all(center_array >= -1.0) and np.all(center_array <= 2.0)

    def test_physical_spreads_computation(self, mock_nscf_result, simple_crystal):
        """Test physical spreads computation."""
        w = Wannier(mock_nscf_result, n_wann=4, crystal=simple_crystal)

        result = w.compute(max_iter=5, verbose=False)
        centers = w.compute_wannier_centers(result.U_matrix)

        total_spread, spreads = w.compute_spreads_physical(centers)

        # Check spreads
        assert total_spread >= 0
        assert len(spreads) == w.n_kpts * w.n_wann


class TestWannierIO:
    """Test Wannier HDF5 I/O."""

    def test_save_wannier_to_hdf5(self, mock_nscf_result, simple_crystal, tmp_path):
        """Test saving Wannier results to HDF5."""
        w = Wannier(mock_nscf_result, n_wann=4, crystal=simple_crystal)
        result = w.compute(max_iter=5, verbose=False)

        output_file = os.path.join(str(tmp_path), 'wannier.h5')
        save_wannier_to_hdf5(output_file, result, crystal=simple_crystal,
                           kgrid=mock_nscf_result.kgrid)

        # Verify file exists
        assert os.path.exists(output_file)

        # Verify structure
        with h5py.File(output_file, 'r') as f:
            # Check attributes
            assert f.attrs['n_wann'] == 4
            assert f.attrs['n_bands'] == 10
            assert 'converged' in f.attrs

            # Check groups
            assert 'U_matrices' in f
            assert 'spreads' in f
            assert 'crystal' in f
            assert 'kpoints' in f

            # Check U matrices
            u_grp = f['U_matrices']
            assert len(u_grp) == 8  # 2x2x2 k-grid

    def test_read_wannier_from_hdf5(self, mock_nscf_result, simple_crystal, tmp_path):
        """Test reading Wannier results from HDF5."""
        w = Wannier(mock_nscf_result, n_wann=4, crystal=simple_crystal)
        result = w.compute(max_iter=5, verbose=False)

        output_file = os.path.join(str(tmp_path), 'wannier.h5')
        save_wannier_to_hdf5(output_file, result, crystal=simple_crystal,
                           kgrid=mock_nscf_result.kgrid)

        # Read back
        data = read_wannier_from_hdf5(output_file)

        # Check data
        assert data['n_wann'] == 4
        assert data['n_bands'] == 10
        assert data['total_spread'] > 0
        assert len(data['U_matrix']) == 8
        assert len(data['spread']) == 8
        assert data['n_kpoints'] == 8

    def test_wannier_centers_in_hdf5(self, mock_nscf_result, simple_crystal, tmp_path):
        """Test saving and loading Wannier centers."""
        w = Wannier(mock_nscf_result, n_wann=4, crystal=simple_crystal)
        result = w.compute(max_iter=5, verbose=False)
        centers = w.compute_wannier_centers(result.U_matrix)

        output_file = os.path.join(str(tmp_path), 'wannier.h5')
        save_wannier_to_hdf5(output_file, result, wannier_centers=centers,
                           crystal=simple_crystal, kgrid=mock_nscf_result.kgrid)

        # Read back
        data = read_wannier_from_hdf5(output_file)

        # Check centers
        assert 'wannier_centers' in data
        assert len(data['wannier_centers']) == 8


class TestWannierProjection:
    """Test Wannier with initial projection."""

    def test_wannier_with_projection(self, mock_nscf_result, simple_crystal):
        """Test Wannier initialization with projection."""
        w = Wannier(mock_nscf_result, n_wann=4, crystal=simple_crystal)

        # Create initial projection (first 4 bands)
        projection = np.eye(w.n_bands, w.n_wann, dtype=complex)

        result = w.compute(max_iter=5, projection=projection, verbose=False)

        # Check result
        assert result.n_wann == 4
        assert len(result.U_matrix) == w.n_kpts


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
