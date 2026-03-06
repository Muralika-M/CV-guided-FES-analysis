#!/usr/bin/env python3
"""
Run the minimal main-plot scripts using local defaults.
"""

from pathlib import Path
import subprocess
import sys


def main():
    package_root = Path(__file__).resolve().parents[1]
    analysis_dir = package_root / "analysis"
    scripts = [
        "threshold_scan.py",
        "hlda_scatter_grid.py",
        "residue_correlation_heatmaps.py",
        "residue_correlation_scatters.py",
    ]
    for name in scripts:
        script_path = analysis_dir / name
        print(f"Running {name}")
        subprocess.run([sys.executable, str(script_path)], check=True)


if __name__ == "__main__":
    main()
