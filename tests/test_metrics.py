import importlib

import numpy as np
import pytest


def _api(name):
    candidates = (
        "talbot.metrics",
        "talbot",
        "optics.metrics",
        "metrics",
        "src.metrics",
        "optics",
        "optika.metrics",
        "optika",
        "main",
    )
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, name):
            return getattr(module, name)
    pytest.fail(f"Cannot import API function {name!r}")


normalized_correlation = _api("normalized_correlation")
relative_l2_error = _api("relative_l2_error")
total_power = _api("total_power")
best_shift_correlation = _api("best_shift_correlation")


def _corr_value(result):
    if isinstance(result, tuple):
        for item in result:
            if np.isscalar(item) and -1e-12 <= item <= 1 + 1e-12:
                return float(item)
    if isinstance(result, dict):
        return result.get("correlation", result.get("corr"))
    return result


def _shift_value(result):
    if isinstance(result, tuple):
        for item in result:
            if np.isscalar(item) and float(item).is_integer() and abs(item) > 1:
                return int(item)
    if isinstance(result, dict):
        return result.get("shift")
    pytest.fail("best_shift_correlation must return a shift as tuple or dict")


def test_normalized_correlation_is_one_for_scaled_global_phase():
    x = np.linspace(-1.0, 1.0, 64)
    u = np.exp(-x**2 / 0.2**2) * np.exp(1j * 0.7 * x)
    v = 2.5 * np.exp(1j * 1.2) * u

    assert normalized_correlation(u, v) == pytest.approx(1.0, abs=1e-14)


def test_relative_l2_error_basic_cases():
    u = np.array([1.0, 2.0, 3.0])

    assert relative_l2_error(u, u) == pytest.approx(0.0)
    assert relative_l2_error(u, 2 * u) == pytest.approx(0.5)


def test_total_power_integrates_intensity_with_dx():
    dx = 0.25e-6
    u = np.array([1 + 1j, 2 - 1j, -1j])

    expected = np.sum(np.abs(u) ** 2) * dx

    assert total_power(u, dx) == pytest.approx(expected)


def test_best_shift_correlation_finds_integer_roll():
    rng = np.random.default_rng(1234)
    u = rng.normal(size=128) + 1j * rng.normal(size=128)
    shift = 17
    v = np.roll(u, shift)

    result = best_shift_correlation(u, v)

    assert _corr_value(result) == pytest.approx(1.0, abs=1e-14)
    assert _shift_value(result) in {shift, shift - u.size, -shift, u.size - shift}
