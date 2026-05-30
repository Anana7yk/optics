"""Transmission functions for one-dimensional periodic gratings."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _wrapped_phase(x: NDArray[np.float64], period: float, shift: float) -> NDArray[np.float64]:
    if period <= 0:
        raise ValueError("period must be positive")
    return np.mod(x - shift, period) / period


def binary_amplitude_grating(
    x: ArrayLike,
    period: float,
    fill: float = 0.5,
    shift: float = 0.0,
) -> NDArray[np.complex128]:
    """Return a binary amplitude grating with unit transmission in open slits."""

    if not 0.0 <= fill <= 1.0:
        raise ValueError("fill must be between 0 and 1")
    x_arr = np.asarray(x, dtype=float)
    phase = _wrapped_phase(x_arr, period, shift)
    return (phase < fill).astype(np.complex128)


def sinusoidal_phase_grating(
    x: ArrayLike,
    period: float,
    phi0: float = np.pi / 2,
) -> NDArray[np.complex128]:
    """Return a pure phase grating exp(i phi0 cos(2 pi x / period))."""

    if period <= 0:
        raise ValueError("period must be positive")
    x_arr = np.asarray(x, dtype=float)
    phase = phi0 * np.cos(2 * np.pi * x_arr / period)
    return np.exp(1j * phase)


def sinusoidal_amplitude_grating(
    x: ArrayLike,
    period: float,
    modulation: float = 0.3,
) -> NDArray[np.complex128]:
    """Return t(x)=1+m cos(2 pi x / period), useful for analytic checks."""

    if period <= 0:
        raise ValueError("period must be positive")
    if abs(modulation) > 1:
        raise ValueError("|modulation| must not exceed 1")
    x_arr = np.asarray(x, dtype=float)
    return (1.0 + modulation * np.cos(2 * np.pi * x_arr / period)).astype(
        np.complex128
    )


def binary_phase_grating(
    x: ArrayLike,
    period: float,
    fill: float = 0.5,
    phase_step: float = np.pi,
    shift: float = 0.0,
) -> NDArray[np.complex128]:
    """Return a two-level phase grating with phases 0 and phase_step."""

    if not 0.0 <= fill <= 1.0:
        raise ValueError("fill must be between 0 and 1")
    x_arr = np.asarray(x, dtype=float)
    phase = _wrapped_phase(x_arr, period, shift)
    return np.exp(1j * phase_step * (phase >= fill))


def blazed_phase_grating(
    x: ArrayLike,
    period: float,
    phase_depth: float = 2 * np.pi,
    shift: float = 0.0,
) -> NDArray[np.complex128]:
    """Return a sawtooth phase grating exp(i phase_depth x/period)."""

    x_arr = np.asarray(x, dtype=float)
    phase = phase_depth * _wrapped_phase(x_arr, period, shift)
    return np.exp(1j * phase)


def triangular_amplitude_grating(
    x: ArrayLike,
    period: float,
    contrast: float = 1.0,
    shift: float = 0.0,
) -> NDArray[np.complex128]:
    """Return a triangular amplitude profile within each period."""

    if not 0.0 <= contrast <= 1.0:
        raise ValueError("contrast must be between 0 and 1")
    x_arr = np.asarray(x, dtype=float)
    phase = _wrapped_phase(x_arr, period, shift)
    triangle = 1.0 - 2.0 * np.abs(phase - 0.5)
    transmission = 1.0 - contrast + contrast * triangle
    return transmission.astype(np.complex128)


def double_slit_amplitude_grating(
    x: ArrayLike,
    period: float,
    slit_width: float = 0.16,
    separation: float = 0.42,
    shift: float = 0.0,
) -> NDArray[np.complex128]:
    """Return a periodic unit cell containing two narrow amplitude slits."""
    if not 0.0 < slit_width <= 0.5:
        raise ValueError("slit_width must be in (0, 0.5]")
    if not 0.0 < separation < 1.0:
        raise ValueError("separation must be in (0, 1)")
    x_arr = np.asarray(x, dtype=float)
    phase = _wrapped_phase(x_arr, period, shift)
    transmission = np.zeros_like(phase)
    centers = (0.5 - separation / 2, 0.5 + separation / 2)
    for center in centers:
        distance = np.minimum(np.abs(phase - center), 1.0 - np.abs(phase - center))
        transmission = np.maximum(transmission, distance < slit_width / 2)
    return transmission.astype(np.complex128)


def multi_slit_amplitude_grating(
    x: ArrayLike,
    period: float,
    slit_count: int = 3,
    slit_width: float = 0.10,
    shift: float = 0.0,
) -> NDArray[np.complex128]:
    """Return a periodic grating with several identical slits in each period."""

    if not 0.0 < slit_width <= 0.5:
        raise ValueError("slit_width must be in (0, 0.5]")
    if slit_count < 1:
        raise ValueError("slit_count must be positive")
    if slit_width >= 0.9 / slit_count:
        raise ValueError("slit_width is too large for the requested slit_count")
    x_arr = np.asarray(x, dtype=float)
    phase = _wrapped_phase(x_arr, period, shift)
    transmission = np.zeros_like(phase)
    centers = (np.arange(slit_count) + 0.5) / slit_count
    for center in centers:
        distance = np.minimum(np.abs(phase - center), 1.0 - np.abs(phase - center))
        transmission = np.maximum(transmission, distance < slit_width / 2)
    return transmission.astype(np.complex128)


def multiharmonic_phase_grating(
    x: ArrayLike,
    period: float,
    phi0: float = np.pi / 2,
) -> NDArray[np.complex128]:
    """Return a smooth phase grating with several Fourier harmonics."""

    if period <= 0:
        raise ValueError("period must be positive")
    x_arr = np.asarray(x, dtype=float)
    angle = 2 * np.pi * x_arr / period
    phase = phi0 * (
        np.cos(angle)
        + 0.45 * np.cos(2 * angle + 0.6)
        + 0.25 * np.sin(3 * angle - 0.4)
    )
    return np.exp(1j * phase)


def hybrid_amplitude_phase_grating(
    x: ArrayLike,
    period: float,
    modulation: float = 0.35,
    phi0: float = np.pi / 2,
) -> NDArray[np.complex128]:
    """Return a grating with simultaneous sinusoidal amplitude and phase modulation."""

    if abs(modulation) > 1:
        raise ValueError("|modulation| must not exceed 1")
    x_arr = np.asarray(x, dtype=float)
    angle = 2 * np.pi * x_arr / period
    amplitude = 1.0 + modulation * np.cos(angle)
    phase = phi0 * np.sin(angle)
    return (amplitude * np.exp(1j * phase)).astype(np.complex128)
