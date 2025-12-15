"""
Exchange-correlation functional module using pylibxc.
"""

import numpy as np
from typing import Tuple, Optional

try:
    import pylibxc
    HAS_PYLIBXC = True
except ImportError:
    HAS_PYLIBXC = False


class XCFunctional:
    """
    Exchange-correlation functional wrapper using pylibxc.

    Supports LDA and GGA functionals.
    """

    def __init__(self, xc_name: str = 'LDA_X+LDA_C_PZ'):
        """
        Initialize XC functional.

        Args:
            xc_name: Name of XC functional. Examples:
                - 'LDA_X+LDA_C_PZ' : LDA (Perdew-Zunger)
                - 'LDA_X+LDA_C_PW' : LDA (Perdew-Wang)
                - 'GGA_X_PBE+GGA_C_PBE' : PBE GGA
                - 'LDA' : shortcut for LDA_X+LDA_C_PZ
                - 'PBE' : shortcut for GGA PBE
        """
        if not HAS_PYLIBXC:
            raise ImportError("pylibxc is required for XC functionals. "
                              "Install with: pip install pylibxc")

        # Handle shortcuts
        xc_name = self._resolve_shortcut(xc_name)

        # Parse functional names
        self.xc_name = xc_name
        self.functionals = []

        for name in xc_name.split('+'):
            name = name.strip()
            func = pylibxc.LibXCFunctional(name, "unpolarized")
            self.functionals.append(func)

        # Determine if GGA
        self.is_gga = any('GGA' in name.upper() for name in xc_name.split('+'))

    def _resolve_shortcut(self, name: str) -> str:
        """Resolve shortcut names to full functional names."""
        shortcuts = {
            'LDA': 'LDA_X+LDA_C_PZ',
            'PZ': 'LDA_X+LDA_C_PZ',
            'PW': 'LDA_X+LDA_C_PW',
            'PBE': 'GGA_X_PBE+GGA_C_PBE',
            'PBESOL': 'GGA_X_PBE_SOL+GGA_C_PBE_SOL',
        }
        return shortcuts.get(name.upper(), name)

    def compute(self, rho: np.ndarray,
                sigma: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Compute XC energy density and potential.

        Args:
            rho: Electron density (n_points,)
            sigma: |grad(rho)|^2 for GGA (n_points,), None for LDA

        Returns:
            (exc, vxc, vsigma) where:
                exc: XC energy density per electron (n_points,)
                vxc: XC potential dE/drho (n_points,)
                vsigma: dE/dsigma for GGA (n_points,) or None for LDA
        """
        # Ensure positive density
        rho_safe = np.maximum(rho, 1e-20)

        # Reshape for pylibxc (expects 2D array with shape (n_points, n_spin))
        inp = {"rho": rho_safe.reshape(-1, 1)}

        if self.is_gga and sigma is not None:
            inp["sigma"] = sigma.reshape(-1, 1)

        # Accumulate contributions from all functionals
        exc_total = np.zeros_like(rho_safe)
        vxc_total = np.zeros_like(rho_safe)
        vsigma_total = np.zeros_like(rho_safe) if self.is_gga else None

        for func in self.functionals:
            ret = func.compute(inp)

            # Energy density (per unit volume, need to convert to per electron)
            if "zk" in ret:
                exc_total += ret["zk"].flatten()

            # Potential
            if "vrho" in ret:
                vxc_total += ret["vrho"].flatten()

            # GGA sigma derivative
            if self.is_gga and "vsigma" in ret:
                vsigma_total += ret["vsigma"].flatten()

        return exc_total, vxc_total, vsigma_total

    def get_exc_energy(self, rho: np.ndarray, volume: float,
                       sigma: np.ndarray = None) -> float:
        """
        Compute total XC energy.

        E_xc = integral[rho(r) * exc(rho(r)) dr]

        Args:
            rho: Electron density on real-space grid
            volume: Cell volume
            sigma: |grad(rho)|^2 for GGA

        Returns:
            Total XC energy in Hartree
        """
        exc, _, _ = self.compute(rho, sigma)

        # Integration weight
        dv = volume / len(rho)

        # E_xc = sum[rho * exc * dv]
        return np.sum(rho * exc) * dv

    def get_vxc_potential(self, rho: np.ndarray,
                          sigma: np.ndarray = None) -> np.ndarray:
        """
        Compute XC potential in real space.

        For LDA: V_xc = d(rho*exc)/d(rho)
        For GGA: V_xc = d(rho*exc)/d(rho) - div[d(rho*exc)/d(grad_rho)]

        Note: For GGA, the gradient term needs special handling.
        This function returns the LDA-like part only.

        Args:
            rho: Electron density
            sigma: |grad(rho)|^2 for GGA

        Returns:
            XC potential on real-space grid
        """
        _, vxc, vsigma = self.compute(rho, sigma)
        return vxc


class SimpleLDA:
    """
    Simple LDA implementation as fallback when pylibxc is not available.

    Uses Perdew-Zunger parametrization for correlation.
    """

    def __init__(self):
        self.is_gga = False
        self.xc_name = 'Simple LDA (PZ)'

    def compute(self, rho: np.ndarray,
                sigma: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, None]:
        """Compute LDA exchange-correlation."""
        rho_safe = np.maximum(rho, 1e-20)

        # Exchange (Slater)
        ex = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0) * rho_safe ** (1.0 / 3.0)
        vx = (4.0 / 3.0) * ex

        # Correlation (Perdew-Zunger)
        rs = (3.0 / (4.0 * np.pi * rho_safe)) ** (1.0 / 3.0)

        # High density (rs < 1)
        A = 0.0311
        B = -0.048
        C = 0.002
        D = -0.0116

        ec_high = A * np.log(rs) + B + C * rs * np.log(rs) + D * rs
        vc_high = A * np.log(rs) + (B - A / 3.0) + (2.0 / 3.0) * C * rs * np.log(rs) + (2 * D - C) / 3.0 * rs

        # Low density (rs >= 1)
        gamma = -0.1423
        beta1 = 1.0529
        beta2 = 0.3334

        sqrt_rs = np.sqrt(rs)
        denom = 1.0 + beta1 * sqrt_rs + beta2 * rs

        ec_low = gamma / denom
        vc_low = ec_low * (1.0 + (7.0 / 6.0) * beta1 * sqrt_rs + (4.0 / 3.0) * beta2 * rs) / denom

        # Combine
        mask = rs < 1.0
        ec = np.where(mask, ec_high, ec_low)
        vc = np.where(mask, vc_high, vc_low)

        exc = ex + ec
        vxc = vx + vc

        return exc, vxc, None

    def get_exc_energy(self, rho: np.ndarray, volume: float,
                       sigma: np.ndarray = None) -> float:
        """Compute total XC energy."""
        exc, _, _ = self.compute(rho)
        dv = volume / len(rho)
        return np.sum(rho * exc) * dv

    def get_vxc_potential(self, rho: np.ndarray,
                          sigma: np.ndarray = None) -> np.ndarray:
        """Compute XC potential."""
        _, vxc, _ = self.compute(rho)
        return vxc


def get_xc_functional(name: str = 'LDA'):
    """
    Get an XC functional by name.

    Args:
        name: Functional name ('LDA', 'PBE', etc.)

    Returns:
        XCFunctional or SimpleLDA instance
    """
    if HAS_PYLIBXC:
        return XCFunctional(name)
    else:
        if 'GGA' in name.upper() or 'PBE' in name.upper():
            raise ImportError("GGA functionals require pylibxc")
        print("Warning: pylibxc not available, using simple LDA implementation")
        return SimpleLDA()
