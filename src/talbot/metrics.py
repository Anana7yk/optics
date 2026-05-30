"""Numerical metrics for comparing propagated fields and intensities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_float_vector(values: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(values)
    if np.iscomplexobj(arr):
        arr = np.abs(arr) ** 2
    return np.asarray(arr, dtype=float).ravel()


def normalized_correlation(a: ArrayLike, b: ArrayLike) -> float:
    """Pearson correlation of two real arrays; complex inputs compare intensities."""

    av = _as_float_vector(a)
    bv = _as_float_vector(b)
    if av.shape != bv.shape:
        raise ValueError("arrays must have equal shapes")
    av = av - np.mean(av)
    bv = bv - np.mean(bv)
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    if denom == 0:
        return 1.0 if np.allclose(av, bv) else 0.0
    return float(np.real(np.vdot(av, bv)) / denom)


def relative_l2_error(a: ArrayLike, b: ArrayLike) -> float:
    """Return ||a-b||_2 / ||b||_2."""

    av = np.asarray(a)
    bv = np.asarray(b)
    if av.shape != bv.shape:
        raise ValueError("arrays must have equal shapes")
    denom = np.linalg.norm(bv)
    if denom == 0:
        return float(np.linalg.norm(av))
    return float(np.linalg.norm(av - bv) / denom)


def total_power(u: ArrayLike, dx: float) -> float:
    """Approximate integral of |u|^2 over x."""

    if dx <= 0:
        raise ValueError("dx must be positive")
    field = np.asarray(u)
    return float(np.sum(np.abs(field) ** 2) * dx)


def best_shift_correlation(a: ArrayLike, b: ArrayLike) -> tuple[int, float]:
    """Return the integer circular shift of b that maximizes correlation with a."""

    av = _as_float_vector(a)
    bv = _as_float_vector(b)
    if av.shape != bv.shape:
        raise ValueError("arrays must have equal shapes")

    av0 = av - np.mean(av)
    bv0 = bv - np.mean(bv)
    denom = np.linalg.norm(av0) * np.linalg.norm(bv0)
    if denom == 0:
        return 0, 1.0 if np.allclose(av, bv) else 0.0

    corr = np.fft.ifft(np.fft.fft(av0) * np.conj(np.fft.fft(bv0))).real / denom
    index = int(np.argmax(corr))
    shift = index if index <= av.size // 2 else index - av.size
    return shift, float(corr[index])
