"""
Pseudopotential reader for ONCVPSP UPF format.

Supports UPF v2 format from ONCVPSP (norm-conserving pseudopotentials).
"""

import numpy as np
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import re
import os


@dataclass
class NonlocalProjector:
    """Nonlocal projector (beta function)."""
    l: int  # Angular momentum
    index: int  # Projector index for this l
    radial_grid: np.ndarray
    values: np.ndarray
    dij: float = 0.0  # D_ij coefficient


@dataclass
class Pseudopotential:
    """
    Norm-conserving pseudopotential from ONCVPSP.

    Stores local potential, nonlocal projectors, and atomic information.
    """
    symbol: str
    z_valence: float  # Number of valence electrons
    z_atom: int  # Atomic number

    # Radial grid
    radial_grid: np.ndarray
    rab: np.ndarray  # dr for integration

    # Local potential V_loc(r) in Hartree
    v_local: np.ndarray

    # Nonlocal projectors
    projectors: List[NonlocalProjector] = field(default_factory=list)

    # D_ij matrix for nonlocal PP
    dij_matrix: np.ndarray = None

    # Core charge density (for NLCC)
    rho_core: np.ndarray = None

    # Atomic valence charge
    rho_atom: np.ndarray = None

    # Maximum angular momentum
    lmax: int = 0

    def get_vloc_of_g(self, g: np.ndarray, volume: float) -> np.ndarray:
        """
        Compute local potential in G-space using spherical Bessel transform.

        V_loc(G) = (4*pi/V) * integral[r^2 * V_loc(r) * sin(Gr)/(Gr) dr]

        For G=0, use a separate treatment with erfc-screened potential.

        Args:
            g: G-vector magnitudes (nG,)
            volume: Cell volume

        Returns:
            V_loc(G) for each G-vector
        """
        vloc_g = np.zeros(len(g), dtype=np.complex128)

        r = self.radial_grid
        dr = self.rab
        vloc_r = self.v_local

        # For numerical stability, we integrate V_loc(r) + Z/r and subtract Z/r analytically
        # V_loc(G) = V_loc_short(G) - 4*pi*Z/(V*G^2) for G != 0

        for ig, gmag in enumerate(g):
            if gmag < 1e-10:
                # G = 0: integrate r^2 * V_loc(r) * dr
                # This needs careful handling - use limit of sin(Gr)/Gr -> 1
                integrand = r**2 * vloc_r
                vloc_g[ig] = 4 * np.pi / volume * np.trapz(integrand, r)
            else:
                # Spherical Bessel j0(Gr) = sin(Gr)/(Gr)
                sinc_gr = np.sinc(gmag * r / np.pi)  # numpy sinc includes pi
                integrand = r**2 * vloc_r * sinc_gr
                vloc_g[ig] = 4 * np.pi / volume * np.trapz(integrand, r)

        return vloc_g.real

    def get_projector_of_g(self, proj_idx: int, g: np.ndarray) -> np.ndarray:
        """
        Compute projector in G-space using spherical Bessel transform.

        beta_l(G) = integral[r^2 * beta_l(r) * j_l(Gr) dr]

        Args:
            proj_idx: Index of projector
            g: G-vector magnitudes

        Returns:
            beta(G) for each G-vector
        """
        proj = self.projectors[proj_idx]
        r = proj.radial_grid
        beta_r = proj.values
        l = proj.l

        beta_g = np.zeros(len(g), dtype=np.complex128)

        for ig, gmag in enumerate(g):
            if gmag < 1e-10:
                if l == 0:
                    # j_0(0) = 1
                    integrand = r**2 * beta_r
                    beta_g[ig] = np.trapz(integrand, r)
                else:
                    # j_l(0) = 0 for l > 0
                    beta_g[ig] = 0.0
            else:
                jl = spherical_bessel_j(l, gmag * r)
                integrand = r**2 * beta_r * jl
                beta_g[ig] = np.trapz(integrand, r)

        return beta_g

    def get_structure_factor_coeffs(self, g: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """
        Get structure factor for atom at position tau.

        S(G) = exp(-i G . tau)

        Args:
            g: G-vectors (nG, 3) in Cartesian
            tau: Atomic position in Cartesian

        Returns:
            Structure factor (nG,)
        """
        return np.exp(-1j * g @ tau)


def spherical_bessel_j(l: int, x: np.ndarray) -> np.ndarray:
    """
    Spherical Bessel function of the first kind.

    j_l(x) for l = 0, 1, 2, 3
    """
    # Handle x = 0 carefully
    result = np.zeros_like(x)
    mask = np.abs(x) > 1e-10

    if l == 0:
        result[mask] = np.sin(x[mask]) / x[mask]
        result[~mask] = 1.0
    elif l == 1:
        x_safe = x[mask]
        result[mask] = np.sin(x_safe) / x_safe**2 - np.cos(x_safe) / x_safe
        result[~mask] = 0.0
    elif l == 2:
        x_safe = x[mask]
        result[mask] = ((3 / x_safe**2 - 1) * np.sin(x_safe) / x_safe
                        - 3 * np.cos(x_safe) / x_safe**2)
        result[~mask] = 0.0
    elif l == 3:
        x_safe = x[mask]
        result[mask] = ((15 / x_safe**3 - 6 / x_safe) * np.sin(x_safe) / x_safe
                        - (15 / x_safe**2 - 1) * np.cos(x_safe) / x_safe)
        result[~mask] = 0.0
    else:
        # Use scipy for higher l
        from scipy.special import spherical_jn
        result = spherical_jn(l, x)

    return result


class UPFReader:
    """
    Reader for UPF (Unified Pseudopotential Format) files.

    Supports UPF version 2 format from ONCVPSP.
    """

    def __init__(self, filepath: str):
        """
        Initialize UPF reader.

        Args:
            filepath: Path to UPF file
        """
        self.filepath = filepath
        self.pp = None

    def read(self) -> Pseudopotential:
        """
        Read and parse UPF file.

        Returns:
            Pseudopotential object
        """
        with open(self.filepath, 'r') as f:
            content = f.read()

        # Check UPF version
        if '<UPF version="2.0.1">' in content or '<UPF version="2' in content:
            return self._read_upf_v2(content)
        else:
            return self._read_upf_v1(content)

    def _read_upf_v2(self, content: str) -> Pseudopotential:
        """Read UPF version 2 format (XML-like)."""
        # Parse as XML
        # UPF v2 wraps data in XML tags

        # Extract header info
        header_match = re.search(r'<PP_HEADER(.*?)/>', content, re.DOTALL)
        if header_match:
            header = header_match.group(1)
        else:
            header_match = re.search(r'<PP_HEADER>(.*?)</PP_HEADER>', content, re.DOTALL)
            header = header_match.group(1) if header_match else ""

        # Parse header attributes
        symbol = self._extract_attr(header, 'element', 'X').strip()
        z_valence = float(self._extract_attr(header, 'z_valence', '0'))
        lmax = int(self._extract_attr(header, 'l_max', '0'))
        mesh_size = int(self._extract_attr(header, 'mesh_size', '0'))
        n_proj = int(self._extract_attr(header, 'number_of_proj', '0'))

        # Get atomic number from element
        from .crystal import ATOMIC_NUMBERS
        z_atom = ATOMIC_NUMBERS.get(symbol, 0)

        # Read radial mesh
        r = self._extract_data(content, 'PP_R')
        rab = self._extract_data(content, 'PP_RAB')

        # Read local potential (in Ry, convert to Ha)
        vloc = self._extract_data(content, 'PP_LOCAL') / 2.0

        # Read nonlocal projectors
        projectors = []
        dij_data = self._extract_data(content, 'PP_DIJ')

        for i in range(1, n_proj + 1):
            beta_data = self._extract_data(content, f'PP_BETA.{i}')
            if beta_data is None:
                beta_data = self._extract_data(content, f'PP_BETA{i}')

            # Get angular momentum from header
            beta_header = re.search(
                rf'<PP_BETA\.{i}(.*?)>', content, re.DOTALL)
            if beta_header:
                l = int(self._extract_attr(beta_header.group(1), 'angular_momentum', '0'))
            else:
                l = 0

            proj = NonlocalProjector(
                l=l,
                index=i,
                radial_grid=r,
                values=beta_data / 2.0 if beta_data is not None else np.zeros_like(r)
            )
            projectors.append(proj)

        # Parse D_ij matrix
        dij_matrix = None
        if dij_data is not None and n_proj > 0:
            # D_ij is stored as a flat array, reshape to matrix
            # Convert from Ry to Ha
            dij_matrix = dij_data.reshape((n_proj, n_proj)) / 2.0

        # Read core charge if present (NLCC)
        rho_core = self._extract_data(content, 'PP_NLCC')

        # Read atomic charge
        rho_atom = self._extract_data(content, 'PP_RHOATOM')

        return Pseudopotential(
            symbol=symbol,
            z_valence=z_valence,
            z_atom=z_atom,
            radial_grid=r,
            rab=rab,
            v_local=vloc,
            projectors=projectors,
            dij_matrix=dij_matrix,
            rho_core=rho_core,
            rho_atom=rho_atom,
            lmax=lmax
        )

    def _read_upf_v1(self, content: str) -> Pseudopotential:
        """Read UPF version 1 format."""
        # Simpler format with data blocks between tags

        lines = content.split('\n')

        # Find sections
        sections = {}
        current_section = None
        current_data = []

        for line in lines:
            if '<PP_' in line and '/>' not in line:
                # Start of section
                match = re.search(r'<(PP_\w+)', line)
                if match:
                    current_section = match.group(1)
                    current_data = []
            elif '</PP_' in line:
                # End of section
                if current_section:
                    sections[current_section] = '\n'.join(current_data)
                current_section = None
            elif current_section:
                current_data.append(line)

        # Parse header
        header = sections.get('PP_HEADER', '')
        header_lines = header.strip().split('\n')

        symbol = header_lines[0].split()[0] if header_lines else 'X'
        z_valence = 4.0  # Default

        for line in header_lines:
            if 'Z valence' in line or 'z_valence' in line.lower():
                z_valence = float(line.split()[0])
                break

        from .crystal import ATOMIC_NUMBERS
        z_atom = ATOMIC_NUMBERS.get(symbol.strip(), 0)

        # Parse mesh
        r = self._parse_data_block(sections.get('PP_R', ''))
        rab = self._parse_data_block(sections.get('PP_RAB', ''))

        # Parse local potential
        vloc = self._parse_data_block(sections.get('PP_LOCAL', '')) / 2.0

        # Simplified: no nonlocal for v1 in this basic implementation
        projectors = []
        dij_matrix = None

        return Pseudopotential(
            symbol=symbol.strip(),
            z_valence=z_valence,
            z_atom=z_atom,
            radial_grid=r,
            rab=rab,
            v_local=vloc,
            projectors=projectors,
            dij_matrix=dij_matrix,
            lmax=0
        )

    def _extract_attr(self, text: str, attr: str, default: str = '') -> str:
        """Extract attribute value from XML-like string."""
        pattern = rf'{attr}\s*=\s*"([^"]*)"'
        match = re.search(pattern, text)
        return match.group(1) if match else default

    def _extract_data(self, content: str, tag: str) -> Optional[np.ndarray]:
        """Extract numerical data from a UPF section."""
        # Try with dot notation first (PP_BETA.1)
        pattern = rf'<{tag}[^>]*>(.*?)</{tag}>'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

        if match:
            data_str = match.group(1)
            return self._parse_data_block(data_str)

        return None

    def _parse_data_block(self, data_str: str) -> np.ndarray:
        """Parse a block of numerical data."""
        # Remove any XML comments or attributes
        data_str = re.sub(r'<[^>]+>', '', data_str)

        # Split and convert to floats
        values = []
        for token in data_str.split():
            try:
                # Handle Fortran-style D notation
                token = token.replace('D', 'E').replace('d', 'e')
                values.append(float(token))
            except ValueError:
                continue

        return np.array(values)


def read_upf(filepath: str) -> Pseudopotential:
    """
    Convenience function to read a UPF file.

    Args:
        filepath: Path to UPF file

    Returns:
        Pseudopotential object
    """
    reader = UPFReader(filepath)
    return reader.read()


def download_oncvpsp_pp(symbol: str, xc: str = 'PBE',
                        output_dir: str = '.') -> str:
    """
    Download pseudopotential from ONCVPSP library (pseudo-dojo or similar).

    This is a placeholder - in practice you'd download from a repository.

    Args:
        symbol: Element symbol
        xc: Exchange-correlation functional
        output_dir: Output directory

    Returns:
        Path to downloaded file
    """
    # For now, just return expected path
    filename = f"{symbol}.upf"
    return os.path.join(output_dir, filename)
