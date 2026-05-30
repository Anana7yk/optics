"""Angular-spectrum propagation models for the slowly varying field envelope."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _spatial_frequencies(n: int, dx: float) -> NDArray[np.float64]:
    if n <= 0:
        raise ValueError("field must contain at least one sample")
    if dx <= 0:
        raise ValueError("dx must be positive")
    return 2 * np.pi * np.fft.fftfreq(n, d=dx)


def propagate_paraxial(
    u0: ArrayLike,
    dx: float,
    wavelength: float,
    z: float,
) -> NDArray[np.complex128]:
    """Propagate a scalar envelope with the paraxial angular-spectrum propagator."""

    if wavelength <= 0:
        raise ValueError("wavelength must be positive")
    u = np.asarray(u0, dtype=np.complex128)
    kx = _spatial_frequencies(u.size, dx)
    k = 2 * np.pi / wavelength
    transfer = np.exp(-1j * z * kx**2 / (2 * k))
    return np.fft.ifft(np.fft.fft(u) * transfer)


def propagate_exact(
    u0: ArrayLike,
    dx: float,
    wavelength: float,
    z: float,
    keep_evanescent: bool = True,
) -> NDArray[np.complex128]:
    """Propagate a scalar envelope with the exact angular-spectrum propagator."""

    if wavelength <= 0:
        raise ValueError("wavelength must be positive")
    u = np.asarray(u0, dtype=np.complex128)
    kx = _spatial_frequencies(u.size, dx)
    k = 2 * np.pi / wavelength
    kz = np.empty_like(kx, dtype=np.complex128)
    propagating = np.abs(kx) <= k
    kz[propagating] = np.sqrt(k**2 - kx[propagating] ** 2) - k
    if keep_evanescent:
        kz[~propagating] = 1j * np.sqrt(kx[~propagating] ** 2 - k**2) - k
        transfer = np.exp(1j * z * kz)
    else:
        kz[~propagating] = 0.0
        transfer = np.exp(1j * z * kz)
        transfer[~propagating] = 0.0
    return np.fft.ifft(np.fft.fft(u) * transfer)


def propagate_angular_spectrum(
    u0: ArrayLike,
    dx: float,
    wavelength: float,
    z: float,
    model: str = "paraxial",
    keep_evanescent: bool = True,
) -> NDArray[np.complex128]:
    """Dispatch to either the paraxial or exact angular-spectrum model."""

    if model == "paraxial":
        return propagate_paraxial(u0, dx, wavelength, z)
    if model == "exact":
        return propagate_exact(u0, dx, wavelength, z, keep_evanescent=keep_evanescent)
    raise ValueError("model must be 'paraxial' or 'exact'")
