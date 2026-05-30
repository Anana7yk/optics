"""Shared simulation parameters and helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def talbot_length(period: float, wavelength: float) -> float:
    """Return the paraxial Talbot length 2 d^2 / lambda."""

    if period <= 0 or wavelength <= 0:
        raise ValueError("period and wavelength must be positive")
    return 2 * period**2 / wavelength


@dataclass(frozen=True)
class SimulationGrid:
    """Uniform one-dimensional transverse grid and propagation distances."""

    wavelength: float = 532e-9
    period: float = 40e-6
    num_periods: int = 64
    nx: int = 8192
    nz: int = 500

    @property
    def z_talbot(self) -> float:
        return talbot_length(self.period, self.wavelength)

    @property
    def lx(self) -> float:
        return self.num_periods * self.period

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def x(self) -> NDArray[np.float64]:
        return (np.arange(self.nx) - self.nx // 2) * self.dx

    @property
    def z_values(self) -> NDArray[np.float64]:
        return np.linspace(0.0, self.z_talbot, self.nz)


def default_grid() -> SimulationGrid:
    """Return the recommended grid from the project specification."""

    return SimulationGrid()
