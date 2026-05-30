import importlib

import numpy as np
import pytest


def _api(name):
    candidates = (
        "talbot.gratings",
        "talbot.propagation",
        "talbot.metrics",
        "talbot",
        "optics.gratings",
        "optics.propagation",
        "optics.metrics",
        "gratings",
        "propagation",
        "metrics",
        "src.gratings",
        "src.propagation",
        "src.metrics",
        "optics",
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


def _remove_global_phase(reference, value):
    phase = np.vdot(reference, value)
    if np.abs(phase) == 0:
        return value
    return value * np.exp(-1j * np.angle(phase))


sinusoidal_amplitude_grating = _api("sinusoidal_amplitude_grating")
propagate_paraxial = _api("propagate_paraxial")
normalized_correlation = _api("normalized_correlation")
relative_l2_error = _api("relative_l2_error")


def _talbot_case(samples_per_period=32, n_periods=64):
    wavelength = 633e-9
    period = 20e-6
    dx = period / samples_per_period
    n = n_periods * samples_per_period
    x = (np.arange(n) - n / 2) * dx
    u0 = sinusoidal_amplitude_grating(x, period, modulation=0.25)
    z_talbot = 2 * period**2 / wavelength
    return u0, dx, wavelength, period, z_talbot


def test_grating_revival_at_talbot_distance():
    u0, dx, wavelength, _period, z_talbot = _talbot_case()

    uz = propagate_paraxial(u0, dx, wavelength, z_talbot)
    uz = _remove_global_phase(u0, uz)

    assert normalized_correlation(u0, uz) > 0.999999
    assert relative_l2_error(u0, uz) < 1e-12


def test_half_talbot_distance_is_shifted_by_half_period():
    samples_per_period = 32
    u0, dx, wavelength, _period, z_talbot = _talbot_case(
        samples_per_period=samples_per_period
    )
    expected = np.roll(u0, samples_per_period // 2)

    uz = propagate_paraxial(u0, dx, wavelength, z_talbot / 2)
    uz = _remove_global_phase(expected, uz)

    assert normalized_correlation(expected, uz) > 0.999999
    assert relative_l2_error(expected, uz) < 1e-12
