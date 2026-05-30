import importlib

import numpy as np
import pytest


def _api(name):
    candidates = (
        "talbot.propagation",
        "talbot.metrics",
        "talbot",
        "optics.propagation",
        "optics.metrics",
        "propagation",
        "metrics",
        "src.propagation",
        "src.metrics",
        "optics",
        "optika.propagation",
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


def _remove_global_phase(reference, value):
    phase = np.vdot(reference, value)
    if np.abs(phase) == 0:
        return value
    return value * np.exp(-1j * np.angle(phase))


propagate_paraxial = _api("propagate_paraxial")
propagate_exact = _api("propagate_exact")
propagate_angular_spectrum = _api("propagate_angular_spectrum")
total_power = _api("total_power")
relative_l2_error = _api("relative_l2_error")


@pytest.mark.parametrize(
    "propagator",
    [propagate_paraxial, propagate_exact, propagate_angular_spectrum],
)
def test_constant_field_is_conserved_up_to_global_phase(propagator):
    n = 256
    dx = 0.5e-6
    wavelength = 633e-9
    z = 0.8e-3
    u0 = np.full(n, 1.7 - 0.4j, dtype=complex)

    uz = propagator(u0, dx, wavelength, z)
    aligned = _remove_global_phase(u0, uz)

    assert uz.shape == u0.shape
    assert np.allclose(aligned, u0, rtol=0, atol=1e-12)


def test_paraxial_propagation_conserves_total_power():
    wavelength = 633e-9
    period = 20e-6
    samples_per_period = 32
    n_periods = 32
    dx = period / samples_per_period
    n = n_periods * samples_per_period
    x = (np.arange(n) - n / 2) * dx
    z = 1.3e-3
    u0 = 1.0 + 0.35 * np.cos(2 * np.pi * x / period)

    uz = propagate_paraxial(u0.astype(complex), dx, wavelength, z)

    assert total_power(uz, dx) == pytest.approx(total_power(u0, dx), rel=2e-12)


def test_paraxial_grid_refinement_is_consistent():
    wavelength = 633e-9
    period = 18e-6
    length = 32 * period
    z = 0.7e-3

    n_coarse = 1024
    n_fine = 2 * n_coarse
    dx_coarse = length / n_coarse
    dx_fine = length / n_fine
    x_coarse = (np.arange(n_coarse) - n_coarse / 2) * dx_coarse
    x_fine = (np.arange(n_fine) - n_fine / 2) * dx_fine
    u_coarse = 1.0 + 0.25 * np.cos(2 * np.pi * x_coarse / period)
    u_fine = 1.0 + 0.25 * np.cos(2 * np.pi * x_fine / period)

    propagated_coarse = propagate_paraxial(u_coarse, dx_coarse, wavelength, z)
    propagated_fine = propagate_paraxial(u_fine, dx_fine, wavelength, z)[::2]
    propagated_fine = _remove_global_phase(propagated_coarse, propagated_fine)

    assert relative_l2_error(propagated_coarse, propagated_fine) < 2e-3


def test_sinusoidal_amplitude_analytic_solution():
    wavelength = 532e-9
    period = 40e-6
    modulation = 0.3
    samples_per_period = 64
    n_periods = 32
    n = samples_per_period * n_periods
    dx = period / samples_per_period
    x = (np.arange(n) - n // 2) * dx
    z = 0.37 * (2 * period**2 / wavelength)
    u0 = 1.0 + modulation * np.cos(2 * np.pi * x / period)

    numerical = propagate_paraxial(u0.astype(complex), dx, wavelength, z)
    analytical = 1.0 + modulation * np.cos(2 * np.pi * x / period) * np.exp(
        -1j * np.pi * wavelength * z / period**2
    )

    assert relative_l2_error(numerical, analytical) < 2e-12
