"""
K-point generation module with symmetry support using spglib.
"""

import numpy as np
from typing import List, Tuple, Optional
from .crystal import Crystal

try:
    import spglib
    HAS_SPGLIB = True
except ImportError:
    HAS_SPGLIB = False


class KPoints:
    """
    K-point grid generator with symmetry reduction.
    """

    def __init__(self, crystal: Crystal, grid: Tuple[int, int, int],
                 shift: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 use_symmetry: bool = True):
        """
        Initialize k-point grid.

        Args:
            crystal: Crystal structure
            grid: Monkhorst-Pack grid dimensions (nk1, nk2, nk3)
            shift: Grid shift in fractional reciprocal coordinates
            use_symmetry: Whether to use symmetry to reduce k-points
        """
        self.crystal = crystal
        self.grid = tuple(grid)
        self.shift = tuple(shift)
        self.use_symmetry = use_symmetry and HAS_SPGLIB

        self._generate_kpoints()

    def _generate_kpoints(self):
        """Generate k-points with optional symmetry reduction."""
        if self.use_symmetry:
            self._generate_with_symmetry()
        else:
            self._generate_full_grid()

    def _generate_full_grid(self):
        """Generate full Monkhorst-Pack grid without symmetry."""
        nk1, nk2, nk3 = self.grid
        s1, s2, s3 = self.shift

        kpoints_frac = []
        weights = []

        total = nk1 * nk2 * nk3
        weight = 1.0 / total

        for i1 in range(nk1):
            for i2 in range(nk2):
                for i3 in range(nk3):
                    k1 = (2 * i1 - nk1 + 1 + 2 * s1) / (2 * nk1)
                    k2 = (2 * i2 - nk2 + 1 + 2 * s2) / (2 * nk2)
                    k3 = (2 * i3 - nk3 + 1 + 2 * s3) / (2 * nk3)
                    kpoints_frac.append([k1, k2, k3])
                    weights.append(weight)

        self.kpoints_frac = np.array(kpoints_frac)
        self.weights = np.array(weights)
        self.nkpts = len(weights)

        # Convert to Cartesian coordinates
        self.kpoints_cart = self.kpoints_frac @ self.crystal.reciprocal_cell

    def _generate_with_symmetry(self):
        """Generate symmetry-reduced k-point grid using spglib."""
        # Get spglib cell
        cell = self.crystal.get_spglib_cell()

        # Get symmetry operations
        symmetry = spglib.get_symmetry(cell, symprec=1e-5)

        if symmetry is None:
            print("Warning: spglib could not find symmetry, using full grid")
            self._generate_full_grid()
            return

        rotations = symmetry['rotations']

        # Generate full grid first
        nk1, nk2, nk3 = self.grid
        s1, s2, s3 = self.shift

        full_kpoints = []
        for i1 in range(nk1):
            for i2 in range(nk2):
                for i3 in range(nk3):
                    k1 = (2 * i1 - nk1 + 1 + 2 * s1) / (2 * nk1)
                    k2 = (2 * i2 - nk2 + 1 + 2 * s2) / (2 * nk2)
                    k3 = (2 * i3 - nk3 + 1 + 2 * s3) / (2 * nk3)
                    full_kpoints.append([k1, k2, k3])

        full_kpoints = np.array(full_kpoints)

        # Find irreducible k-points
        ir_kpoints, ir_weights = self._find_irreducible_kpoints(
            full_kpoints, rotations)

        self.kpoints_frac = ir_kpoints
        self.weights = ir_weights
        self.nkpts = len(ir_weights)

        # Convert to Cartesian coordinates
        self.kpoints_cart = self.kpoints_frac @ self.crystal.reciprocal_cell

    def _find_irreducible_kpoints(self, kpoints: np.ndarray,
                                   rotations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find irreducible k-points using symmetry operations.

        Args:
            kpoints: Full k-point grid (nk, 3)
            rotations: Rotation matrices from spglib (nsym, 3, 3)

        Returns:
            (irreducible_kpoints, weights)
        """
        nk = len(kpoints)
        nsym = len(rotations)

        # Track which k-points have been assigned to stars
        assigned = np.zeros(nk, dtype=bool)
        ir_kpoints = []
        ir_weights = []

        for ik in range(nk):
            if assigned[ik]:
                continue

            # This is a new irreducible k-point
            k = kpoints[ik]
            ir_kpoints.append(k)

            # Find all equivalent k-points (the star)
            star_size = 0
            for rot in rotations:
                k_rot = rot @ k
                # Bring back to first BZ
                k_rot = k_rot - np.round(k_rot)

                # Find matching k-point
                for jk in range(nk):
                    if assigned[jk]:
                        continue
                    diff = kpoints[jk] - k_rot
                    diff = diff - np.round(diff)
                    if np.linalg.norm(diff) < 1e-8:
                        assigned[jk] = True
                        star_size += 1
                        break

            ir_weights.append(star_size / nk)

        return np.array(ir_kpoints), np.array(ir_weights)

    def get_high_symmetry_path(self, path_str: str = None,
                                npoints: int = 50) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, str]]]:
        """
        Generate k-point path through high-symmetry points.

        Args:
            path_str: Path specification like "GXWLGK" (None for auto)
            npoints: Number of points per segment

        Returns:
            (kpoints_cart, distances, labels)
        """
        # Get high symmetry points from spglib
        if HAS_SPGLIB:
            cell = self.crystal.get_spglib_cell()
            path_data = spglib.get_symmetry_dataset(cell, symprec=1e-5)

            if path_data is not None:
                # Get standard high symmetry points based on Bravais lattice
                special_points = self._get_special_points(path_data)
            else:
                special_points = self._get_cubic_special_points()
        else:
            special_points = self._get_cubic_special_points()

        # Default path for FCC/diamond
        if path_str is None:
            path_str = "GXWLGK"

        # Generate path
        kpoints = []
        distances = []
        labels = []

        current_dist = 0.0

        for i, sym in enumerate(path_str):
            if sym not in special_points:
                continue

            k_frac = special_points[sym]
            k_cart = k_frac @ self.crystal.reciprocal_cell

            if i == 0:
                kpoints.append(k_cart)
                distances.append(0.0)
                labels.append((0.0, sym))
            else:
                # Previous point
                prev_sym = path_str[i - 1]
                if prev_sym not in special_points:
                    continue

                k_prev_frac = special_points[prev_sym]
                k_prev_cart = k_prev_frac @ self.crystal.reciprocal_cell

                # Interpolate
                for j in range(1, npoints + 1):
                    t = j / npoints
                    k_interp = k_prev_cart + t * (k_cart - k_prev_cart)
                    kpoints.append(k_interp)

                    dk = k_cart - k_prev_cart
                    current_dist += np.linalg.norm(dk) / npoints
                    distances.append(current_dist)

                labels.append((current_dist, sym))

        return np.array(kpoints), np.array(distances), labels

    def _get_special_points(self, sym_data) -> dict:
        """Get special points based on space group."""
        # For simplicity, use FCC points (covers Si, diamond structures)
        return self._get_cubic_special_points()

    def _get_cubic_special_points(self) -> dict:
        """High symmetry points for FCC Brillouin zone."""
        return {
            'G': np.array([0.0, 0.0, 0.0]),       # Gamma
            'X': np.array([0.5, 0.0, 0.5]),       # X
            'W': np.array([0.5, 0.25, 0.75]),     # W
            'L': np.array([0.5, 0.5, 0.5]),       # L
            'K': np.array([0.375, 0.375, 0.75]),  # K
            'U': np.array([0.625, 0.25, 0.625]),  # U
        }

    def __len__(self):
        return self.nkpts

    def __iter__(self):
        for i in range(self.nkpts):
            yield self.kpoints_cart[i], self.weights[i]
