"""Plot generation helpers for the Talbot-effect report."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .gratings import (
    binary_amplitude_grating,
    binary_phase_grating,
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
from .propagation import propagate_exact, propagate_paraxial
from .simulation import SimulationGrid


FIGURE_NAMES = (
    "talbot_carpet_phase.png",
    "talbot_carpet_amplitude.png",
    "talbot_carpet_sinusoidal_amplitude.png",
    "talbot_carpet_binary_amplitude_narrow.png",
    "talbot_carpet_binary_amplitude_wide.png",
    "talbot_carpet_binary_phase_pi.png",
    "talbot_carpet_blazed_phase.png",
    "talbot_carpet_triangular_amplitude.png",
    "talbot_carpet_double_slit_cell.png",
    "talbot_carpet_three_slit_cell.png",
    "talbot_carpet_five_slit_cell.png",
    "talbot_carpet_seven_slit_cell.png",
    "talbot_carpet_multiharmonic_phase.png",
    "talbot_carpet_hybrid_amplitude_phase.png",
    "self_image_correlation.png",
    "diffraction_spectrum_phase.png",
    "diffraction_spectrum_amplitude.png",
    "paraxial_vs_exact.png",
    "field_slices.png",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "figures"


def _normalize_intensity(intensity: NDArray[np.float64]) -> NDArray[np.float64]:
    max_value = float(np.max(intensity))
    return intensity / max_value if max_value > 0 else intensity


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=max(dpi, 200), bbox_inches="tight")
    plt.close(fig)


def _propagated_intensity(
    field: NDArray[np.complex128],
    grid: SimulationGrid,
    z_values: NDArray[np.float64],
    model: str = "paraxial",
) -> NDArray[np.float64]:
    propagator = propagate_paraxial if model == "paraxial" else propagate_exact
    rows = [
        np.abs(propagator(field, grid.dx, grid.wavelength, float(z))) ** 2
        for z in z_values
    ]
    return np.asarray(rows, dtype=float)


def _plot_carpet(
    field: NDArray[np.complex128],
    grid: SimulationGrid,
    output_dir: Path,
    file_name: str,
    title: str,
    dpi: int,
) -> Path:
    z_values = grid.z_values
    carpet = _normalize_intensity(_propagated_intensity(field, grid, z_values))

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    image = ax.imshow(
        carpet,
        origin="lower",
        aspect="auto",
        extent=(
            grid.x[0] / grid.period,
            grid.x[-1] / grid.period,
            z_values[0] / grid.z_talbot,
            z_values[-1] / grid.z_talbot,
        ),
        cmap="magma",
        vmin=0.0,
    )
    ax.set_title(title)
    ax.set_xlabel(r"$x/d$")
    ax.set_ylabel(r"$z/z_T$")
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("Нормированная интенсивность")

    path = output_dir / file_name
    _save(fig, path, dpi)
    return path


def _additional_carpet_specs(
    x: NDArray[np.float64],
    grid: SimulationGrid,
    phi0: float,
) -> list[tuple[NDArray[np.complex128], str, str]]:
    return [
        (
            sinusoidal_amplitude_grating(x, grid.period, modulation=0.45),
            "talbot_carpet_sinusoidal_amplitude.png",
            "Ковер Талбота: синусоидальная амплитудная решетка",
        ),
        (
            binary_amplitude_grating(x, grid.period, fill=0.25),
            "talbot_carpet_binary_amplitude_narrow.png",
            "Ковер Талбота: узкие амплитудные щели",
        ),
        (
            binary_amplitude_grating(x, grid.period, fill=0.75),
            "talbot_carpet_binary_amplitude_wide.png",
            "Ковер Талбота: широкие амплитудные щели",
        ),
        (
            binary_phase_grating(x, grid.period, fill=0.5, phase_step=np.pi),
            "talbot_carpet_binary_phase_pi.png",
            "Ковер Талбота: бинарная фазовая решетка",
        ),
        (
            blazed_phase_grating(x, grid.period, phase_depth=1.7 * np.pi),
            "talbot_carpet_blazed_phase.png",
            "Ковер Талбота: неполная пилообразная фазовая решетка",
        ),
        (
            triangular_amplitude_grating(x, grid.period, contrast=1.0),
            "talbot_carpet_triangular_amplitude.png",
            "Ковер Талбота: треугольная амплитудная решетка",
        ),
        (
            double_slit_amplitude_grating(x, grid.period, slit_width=0.16, separation=0.42),
            "talbot_carpet_double_slit_cell.png",
            "Ковер Талбота: две щели в периоде",
        ),
        (
            multi_slit_amplitude_grating(x, grid.period, slit_count=3, slit_width=0.13),
            "talbot_carpet_three_slit_cell.png",
            "Ковер Талбота: три щели в периоде",
        ),
        (
            multi_slit_amplitude_grating(x, grid.period, slit_count=5, slit_width=0.075),
            "talbot_carpet_five_slit_cell.png",
            "Ковер Талбота: пять щелей в периоде",
        ),
        (
            multi_slit_amplitude_grating(x, grid.period, slit_count=7, slit_width=0.052),
            "talbot_carpet_seven_slit_cell.png",
            "Ковер Талбота: семь щелей в периоде",
        ),
        (
            multiharmonic_phase_grating(x, grid.period, phi0=phi0),
            "talbot_carpet_multiharmonic_phase.png",
            "Ковер Талбота: многочастотная фазовая решетка",
        ),
        (
            hybrid_amplitude_phase_grating(x, grid.period, modulation=0.35, phi0=phi0),
            "talbot_carpet_hybrid_amplitude_phase.png",
            "Ковер Талбота: смешанная амплитудно-фазовая решетка",
        ),
    ]


def _gallery_grid(grid: SimulationGrid) -> SimulationGrid:
    return SimulationGrid(
        wavelength=grid.wavelength,
        period=grid.period,
        num_periods=min(grid.num_periods, 32),
        nx=min(grid.nx, 2048),
        nz=min(grid.nz, 180),
    )


def _plot_self_image_correlation(
    phase_field: NDArray[np.complex128],
    amplitude_field: NDArray[np.complex128],
    grid: SimulationGrid,
    output_dir: Path,
    dpi: int,
) -> Path:
    z_values = np.linspace(0.0, grid.z_talbot, 260)
    phase_ref = np.abs(phase_field) ** 2
    amplitude_ref = np.abs(amplitude_field) ** 2

    phase_corr = []
    amplitude_corr = []
    for z in z_values:
        phase_i = np.abs(propagate_paraxial(phase_field, grid.dx, grid.wavelength, float(z))) ** 2
        amplitude_i = np.abs(propagate_paraxial(amplitude_field, grid.dx, grid.wavelength, float(z))) ** 2
        phase_corr.append(best_shift_correlation(phase_ref, phase_i)[1])
        amplitude_corr.append(best_shift_correlation(amplitude_ref, amplitude_i)[1])

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    ax.plot(
        z_values / grid.z_talbot,
        amplitude_corr,
        label="Амплитудная решетка",
        color="#b54835",
        linewidth=2.0,
    )
    ax.plot(
        z_values / grid.z_talbot,
        phase_corr,
        label="Фазовая решетка",
        color="#246a73",
        linewidth=2.0,
    )
    ax.set_title("Корреляция самоизображения с учетом сдвига")
    ax.set_xlabel(r"$z/z_T$")
    ax.set_ylabel("Максимальная корреляция")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    path = output_dir / "self_image_correlation.png"
    _save(fig, path, dpi)
    return path


def _plot_diffraction_spectrum(
    field: NDArray[np.complex128],
    grid: SimulationGrid,
    output_dir: Path,
    file_name: str,
    title: str,
    dpi: int,
) -> Path:
    spectrum = np.fft.fftshift(np.fft.fft(field)) / field.size
    normalized_frequency = np.fft.fftshift(np.fft.fftfreq(field.size, d=grid.dx)) * grid.period
    mask = np.abs(normalized_frequency) <= 14

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    markerline, stemlines, baseline = ax.stem(
        normalized_frequency[mask],
        np.abs(spectrum[mask]) ** 2,
        basefmt=" ",
    )
    plt.setp(markerline, color="#2d5f73", markersize=4)
    plt.setp(stemlines, color="#2d5f73", linewidth=1.2)
    plt.setp(baseline, visible=False)
    ax.set_title(title)
    ax.set_xlabel(r"Номер порядка $m = f_x d$")
    ax.set_ylabel(r"$|c_m|^2$")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, max(1.0, float(np.max(np.abs(spectrum[mask]) ** 2)) * 2))
    ax.grid(True, which="both", alpha=0.22)

    path = output_dir / file_name
    _save(fig, path, dpi)
    return path


def _plot_paraxial_vs_exact(
    phase_field: NDArray[np.complex128],
    grid: SimulationGrid,
    output_dir: Path,
    dpi: int,
) -> Path:
    z_values = np.linspace(0.0, grid.z_talbot, 180)
    close_period_grid = SimulationGrid(
        wavelength=grid.wavelength,
        period=4.0 * grid.wavelength,
        num_periods=64,
        nx=grid.nx,
        nz=grid.nz,
    )
    x_close = close_period_grid.x
    close_field = sinusoidal_phase_grating(x_close, close_period_grid.period, phi0=np.pi / 2)
    z_close = np.linspace(0.0, close_period_grid.z_talbot, 180)

    errors_default = []
    power_delta_default = []
    errors_close = []
    for z in z_values:
        paraxial = propagate_paraxial(phase_field, grid.dx, grid.wavelength, float(z))
        exact = propagate_exact(phase_field, grid.dx, grid.wavelength, float(z))
        errors_default.append(relative_l2_error(np.abs(paraxial) ** 2, np.abs(exact) ** 2))
        exact_power = total_power(exact, grid.dx)
        if exact_power > 0:
            power_delta_default.append(abs(total_power(paraxial, grid.dx) - exact_power) / exact_power)
        else:
            power_delta_default.append(0.0)
    for z in z_close:
        paraxial = propagate_paraxial(close_field, close_period_grid.dx, close_period_grid.wavelength, float(z))
        exact = propagate_exact(close_field, close_period_grid.dx, close_period_grid.wavelength, float(z))
        errors_close.append(relative_l2_error(np.abs(paraxial) ** 2, np.abs(exact) ** 2))

    z_check = 0.65 * close_period_grid.z_talbot
    paraxial_slice = np.abs(
        propagate_paraxial(close_field, close_period_grid.dx, close_period_grid.wavelength, z_check)
    ) ** 2
    exact_slice = np.abs(
        propagate_exact(close_field, close_period_grid.dx, close_period_grid.wavelength, z_check)
    ) ** 2
    slice_correlation = normalized_correlation(paraxial_slice, exact_slice)

    fig, (ax_error, ax_slice) = plt.subplots(2, 1, figsize=(8.3, 6.4), sharex=False)
    ax_error.plot(
        z_values / grid.z_talbot,
        errors_default,
        label=rf"$d/\lambda={grid.period / grid.wavelength:.1f}$",
        color="#246a73",
        linewidth=2.0,
    )
    ax_error.plot(
        z_values / grid.z_talbot,
        power_delta_default,
        label="Разность мощности",
        color="#4a4a4a",
        linewidth=1.5,
        linestyle=":",
    )
    ax_error.plot(
        z_close / close_period_grid.z_talbot,
        errors_close,
        label=rf"$d/\lambda={close_period_grid.period / close_period_grid.wavelength:.1f}$",
        color="#b54835",
        linewidth=2.0,
    )
    ax_error.set_title("Параксиальная и точная модели")
    ax_error.set_xlabel(r"$z/z_T$")
    ax_error.set_ylabel(r"Отн. ошибка интенсивности")
    ax_error.grid(True, alpha=0.25)
    ax_error.legend(loc="upper right")

    ax_slice.plot(
        x_close / close_period_grid.period,
        _normalize_intensity(paraxial_slice),
        label="Параксиальная",
        color="#246a73",
        linewidth=1.8,
    )
    ax_slice.plot(
        x_close / close_period_grid.period,
        _normalize_intensity(exact_slice),
        label="Точная",
        color="#b54835",
        linewidth=1.6,
        linestyle="--",
    )
    ax_slice.set_title(rf"Срез при $z/z_T=0.65$, корреляция = {slice_correlation:.3f}")
    ax_slice.set_xlim(-4, 4)
    ax_slice.set_xlabel(r"$x/d$")
    ax_slice.set_ylabel("Норм. интенсивность")
    ax_slice.grid(True, alpha=0.25)
    ax_slice.legend(loc="upper right")

    path = output_dir / "paraxial_vs_exact.png"
    _save(fig, path, dpi)
    return path


def _plot_field_slices(
    phase_field: NDArray[np.complex128],
    amplitude_field: NDArray[np.complex128],
    grid: SimulationGrid,
    output_dir: Path,
    dpi: int,
) -> Path:
    slice_positions = (0.0, 0.25, 0.5, 1.0)
    colors = ("#2d5f73", "#6a994e", "#7b4f9d", "#b54835")

    fig, axes = plt.subplots(2, 1, figsize=(8.3, 6.2), sharex=True)
    for position, color in zip(slice_positions, colors):
        distance = position * grid.z_talbot
        phase_i = np.abs(propagate_paraxial(phase_field, grid.dx, grid.wavelength, distance)) ** 2
        amplitude_i = np.abs(propagate_paraxial(amplitude_field, grid.dx, grid.wavelength, distance)) ** 2
        label = rf"$z/z_T={position:g}$"
        axes[0].plot(grid.x / grid.period, _normalize_intensity(phase_i), color=color, label=label)
        axes[1].plot(grid.x / grid.period, _normalize_intensity(amplitude_i), color=color, label=label)

    axes[0].set_title("Срезы интенсивности: фазовая решетка")
    axes[1].set_title("Срезы интенсивности: амплитудная решетка")
    for ax in axes:
        ax.set_xlim(-4, 4)
        ax.set_ylabel("Норм. интенсивность")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, loc="upper right")
    axes[1].set_xlabel(r"$x/d$")

    path = output_dir / "field_slices.png"
    _save(fig, path, dpi)
    return path


def generate_all_figures(
    output_dir: Path | str = DEFAULT_FIGURES_DIR,
    dpi: int = 220,
    grid: SimulationGrid | None = None,
    fill: float = 0.5,
    phi0: float = np.pi / 2,
    progress: Callable[[Path], None] | None = None,
) -> list[Path]:
    """Generate all required figures and return their paths."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    simulation_grid = grid or SimulationGrid()
    x = simulation_grid.x
    phase_field = sinusoidal_phase_grating(x, simulation_grid.period, phi0=phi0)
    amplitude_field = binary_amplitude_grating(x, simulation_grid.period, fill=fill)
    gallery_grid = _gallery_grid(simulation_grid)
    gallery_x = gallery_grid.x

    generated: list[Path] = []

    def add(path: Path) -> None:
        generated.append(path)
        if progress is not None:
            progress(path)

    add(
        _plot_carpet(
            phase_field,
            simulation_grid,
            output_path,
            "talbot_carpet_phase.png",
            "Ковер Талбота: синусоидальная фазовая решетка",
            dpi,
        )
    )
    add(
        _plot_carpet(
            amplitude_field,
            simulation_grid,
            output_path,
            "talbot_carpet_amplitude.png",
            "Ковер Талбота: бинарная амплитудная решетка",
            dpi,
        )
    )
    add(_plot_self_image_correlation(phase_field, amplitude_field, simulation_grid, output_path, dpi))
    add(
        _plot_diffraction_spectrum(
            phase_field,
            simulation_grid,
            output_path,
            "diffraction_spectrum_phase.png",
            "Спектр пространственных частот: фазовая решетка",
            dpi,
        )
    )
    add(
        _plot_diffraction_spectrum(
            amplitude_field,
            simulation_grid,
            output_path,
            "diffraction_spectrum_amplitude.png",
            "Спектр пространственных частот: амплитудная решетка",
            dpi,
        )
    )
    add(_plot_paraxial_vs_exact(phase_field, simulation_grid, output_path, dpi))
    add(_plot_field_slices(phase_field, amplitude_field, simulation_grid, output_path, dpi))

    for field, file_name, title in _additional_carpet_specs(gallery_x, gallery_grid, phi0):
        add(_plot_carpet(field, gallery_grid, output_path, file_name, title, dpi))
    return generated


__all__ = ["DEFAULT_FIGURES_DIR", "FIGURE_NAMES", "generate_all_figures"]
