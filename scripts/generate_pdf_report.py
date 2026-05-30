"""Compile the LaTeX PDF report for the Talbot-effect project."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_TEX = REPORTS_DIR / "talbot_report.tex"
DEFAULT_OUTPUT = REPORTS_DIR / "talbot_report.pdf"


def compile_report(tex_path: Path = DEFAULT_TEX, output_path: Path = DEFAULT_OUTPUT) -> Path:
    """Compile the LaTeX report with xelatex and return the resulting PDF path."""

    compiler = shutil.which("xelatex")
    if compiler is None:
        raise RuntimeError("xelatex is required to compile the LaTeX report")

    tex_path = tex_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        compiler,
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={output_path.parent}",
        str(tex_path),
    ]

    for _ in range(3):
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    generated = output_path.parent / f"{tex_path.stem}.pdf"
    if generated != output_path:
        generated.replace(output_path)
    for suffix in (".aux", ".log", ".out", ".toc"):
        auxiliary = output_path.parent / f"{tex_path.stem}{suffix}"
        auxiliary.unlink(missing_ok=True)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Собрать LaTeX PDF-отчет по эффекту Талбота.")
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX, help="Путь к .tex-файлу.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Путь к PDF-файлу.")
    args = parser.parse_args()
    output = compile_report(args.tex, args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
