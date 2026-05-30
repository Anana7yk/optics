"""Run all project generation tasks."""

from __future__ import annotations

from generate_figures import main as generate_figures_main


def main() -> int:
    return generate_figures_main()


if __name__ == "__main__":
    raise SystemExit(main())
