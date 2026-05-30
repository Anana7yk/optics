"""Utilities for simulating the Talbot effect in one transverse dimension."""

from .gratings import (
    binary_phase_grating,
    binary_amplitude_grating,
    blazed_phase_grating,
    double_slit_amplitude_grating,
    hybrid_amplitude_phase_grating,
    multi_slit_amplitude_grating,
    multiharmonic_phase_grating,
    sinusoidal_amplitude_grating,
    sinusoidal_phase_grating,
    triangular_amplitude_grating,
)
from .metrics import (
    best_shift_correlation,
    normalized_correlation,
    relative_l2_error,
    total_power,
)
from .propagation import (
    propagate_angular_spectrum,
    propagate_exact,
    propagate_paraxial,
)
from .simulation import SimulationGrid, default_grid, talbot_length

__all__ = [
    "SimulationGrid",
    "best_shift_correlation",
    "binary_amplitude_grating",
    "binary_phase_grating",
    "blazed_phase_grating",
    "default_grid",
    "double_slit_amplitude_grating",
    "hybrid_amplitude_phase_grating",
    "multi_slit_amplitude_grating",
    "multiharmonic_phase_grating",
    "normalized_correlation",
    "propagate_angular_spectrum",
    "propagate_exact",
    "propagate_paraxial",
    "relative_l2_error",
    "sinusoidal_amplitude_grating",
    "sinusoidal_phase_grating",
    "talbot_length",
    "total_power",
    "triangular_amplitude_grating",
]
