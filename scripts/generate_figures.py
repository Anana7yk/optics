"""Command-line entry point for Talbot figure generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from talbot.plotting import DEFAULT_FIGURES_DIR, generate_all_figures
from talbot.simulation import SimulationGrid


def main() -> int:
    parser = argparse.ArgumentParser(description="Сгенерировать иллюстрации эффекта Талбота.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Каталог для PNG-файлов.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Разрешение PNG, не меньше 200 dpi.")
    parser.add_argument("--wavelength", type=float, default=532e-9, help="Длина волны в метрах.")
    parser.add_argument("--period", type=float, default=40e-6, help="Период решетки в метрах.")
    parser.add_argument("--num-periods", type=int, default=64, help="Число периодов в расчетном окне.")
    parser.add_argument("--nx", type=int, default=8192, help="Число точек по x.")
    parser.add_argument("--nz", type=int, default=500, help="Число точек по z для ковров.")
    parser.add_argument("--fill", type=float, default=0.5, help="Коэффициент заполнения амплитудной решетки.")
    parser.add_argument("--phi0", type=float, default=1.5707963267948966, help="Фазовая глубина в радианах.")
    args = parser.parse_args()

    dpi = max(args.dpi, 200)
    grid = SimulationGrid(
        wavelength=args.wavelength,
        period=args.period,
        num_periods=args.num_periods,
        nx=args.nx,
        nz=args.nz,
    )
    generated = generate_all_figures(
        output_dir=args.output_dir,
        dpi=dpi,
        grid=grid,
        fill=args.fill,
        phi0=args.phi0,
        progress=lambda path: print(path, flush=True),
    )
    assert generated
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
