import importlib

import numpy as np
import pytest


def _api(name):
    candidates = (
        "talbot.gratings",
        "talbot",
        "optics.gratings",
        "gratings",
        "src.gratings",
        "optics",
        "optika.gratings",
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


binary_amplitude_grating = _api("binary_amplitude_grating")
binary_phase_grating = _api("binary_phase_grating")
blazed_phase_grating = _api("blazed_phase_grating")
double_slit_amplitude_grating = _api("double_slit_amplitude_grating")
hybrid_amplitude_phase_grating = _api("hybrid_amplitude_phase_grating")
multi_slit_amplitude_grating = _api("multi_slit_amplitude_grating")
multiharmonic_phase_grating = _api("multiharmonic_phase_grating")
sinusoidal_phase_grating = _api("sinusoidal_phase_grating")
sinusoidal_amplitude_grating = _api("sinusoidal_amplitude_grating")
triangular_amplitude_grating = _api("triangular_amplitude_grating")


def test_binary_amplitude_grating_is_binary_and_periodic():
    period = 8e-6
    samples_per_period = 256
    dx = period / samples_per_period
    x = (np.arange(4 * samples_per_period) + 0.5) * dx

    grating = binary_amplitude_grating(x, period, fill=0.25)

    assert grating.shape == x.shape
    assert set(np.unique(grating)).issubset({0, 1})
    assert np.array_equal(
        grating[:samples_per_period],
        grating[samples_per_period : 2 * samples_per_period],
    )
    assert np.mean(grating) == pytest.approx(0.25, abs=1 / samples_per_period)


def test_binary_amplitude_grating_shift_moves_pattern():
    period = 12e-6
    samples_per_period = 48
    dx = period / samples_per_period
    x = (np.arange(4 * samples_per_period) + 0.5) * dx
    shift = period / 4

    base = binary_amplitude_grating(x, period, fill=0.5, shift=0.0)
    shifted = binary_amplitude_grating(x, period, fill=0.5, shift=shift)

    assert np.array_equal(shifted, np.roll(base, samples_per_period // 4))


def test_sinusoidal_amplitude_grating_matches_analytic_profile():
    period = 10e-6
    modulation = 0.3
    x = np.linspace(-period, period, 513)

    grating = sinusoidal_amplitude_grating(x, period, modulation=modulation)
    expected = 1.0 + modulation * np.cos(2 * np.pi * x / period)

    assert grating.shape == x.shape
    assert np.allclose(grating, expected, rtol=0, atol=2e-15)
    assert np.min(grating) == pytest.approx(1.0 - modulation)
    assert np.max(grating) == pytest.approx(1.0 + modulation)


def test_sinusoidal_phase_grating_has_unit_amplitude_and_expected_phase():
    period = 9e-6
    phi0 = np.pi / 3
    x = np.linspace(0.0, period, 129, endpoint=False)

    grating = sinusoidal_phase_grating(x, period, phi0=phi0)
    expected = np.exp(1j * phi0 * np.cos(2 * np.pi * x / period))

    assert grating.shape == x.shape
    assert np.allclose(np.abs(grating), 1.0, atol=2e-15)
    assert np.allclose(grating, expected, atol=2e-15)


def test_additional_phase_gratings_have_unit_amplitude():
    period = 11e-6
    x = np.linspace(0.0, 3 * period, 384, endpoint=False)

    for grating in (
        binary_phase_grating(x, period, phase_step=np.pi),
        blazed_phase_grating(x, period, phase_depth=2 * np.pi),
        multiharmonic_phase_grating(x, period, phi0=1.1),
    ):
        assert grating.shape == x.shape
        assert np.allclose(np.abs(grating), 1.0, atol=2e-15)


def test_additional_amplitude_gratings_are_bounded_and_periodic():
    period = 13e-6
    samples_per_period = 96
    x = np.arange(4 * samples_per_period) * period / samples_per_period

    for grating in (
        triangular_amplitude_grating(x, period, contrast=0.8),
        double_slit_amplitude_grating(x, period, slit_width=0.18, separation=0.45),
        multi_slit_amplitude_grating(x, period, slit_count=5, slit_width=0.08),
        hybrid_amplitude_phase_grating(x, period, modulation=0.25, phi0=0.9),
    ):
        assert grating.shape == x.shape
        assert np.all(np.abs(grating) >= -1e-15)
        assert np.all(np.abs(grating) <= 1.25 + 1e-15)
        assert np.allclose(grating[:samples_per_period], grating[samples_per_period : 2 * samples_per_period])
