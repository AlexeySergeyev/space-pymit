import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


@dataclass(frozen=True)
class PoleScanCandidateResult:
    index: int
    initial_lambda: float
    initial_beta: float
    status: str
    chi_square: Optional[float] = None
    dev: Optional[float] = None
    shadow_percent: Optional[float] = None
    fitted_lambda: Optional[float] = None
    fitted_beta: Optional[float] = None
    fitted_period: Optional[float] = None
    areas_file: Optional[str] = None
    lightcurve_output_file: Optional[str] = None
    param_file: Optional[str] = None
    stdout_log_file: Optional[str] = None
    error: str = ""
    fit_result: dict = field(default_factory=dict)


_SCAN_RESULT_FIELDS = [
    "index",
    "initial_lambda",
    "initial_beta",
    "status",
    "chi_square",
    "dev",
    "shadow_percent",
    "fitted_lambda",
    "fitted_beta",
    "fitted_period",
    "areas_file",
    "lightcurve_output_file",
    "param_file",
    "stdout_log_file",
    "error",
]

_POLE_SCAN_MAP_METRICS = [
    ("chi_square", r"$\chi^2$", ""),
    ("fitted_period", "Period [h]", "period"),
    ("shadow_percent", "Dark facet area [%]", "shadow_percent"),
]


def golden_spiral_g10(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate 2n + 1 approximately even lon/lat pole guesses."""
    if n < 0:
        raise ValueError("n must be non-negative.")

    phi = (1 + np.sqrt(5)) / 2
    i = np.arange(-n, n + 1)
    lat = np.arcsin(2 * i / (2 * n + 1)) * 180 / np.pi
    lon = np.mod(i * phi, 1.0) * 360
    return lon, lat


def scan_result_to_dict(result: PoleScanCandidateResult) -> dict:
    return asdict(result)


def write_scan_results(
    results: list[PoleScanCandidateResult], output_file: str | Path
) -> Path:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SCAN_RESULT_FIELDS)
        writer.writeheader()
        for result in results:
            data = scan_result_to_dict(result)
            writer.writerow({field: data.get(field) for field in _SCAN_RESULT_FIELDS})
    return path


def write_best_result(result: PoleScanCandidateResult, output_file: str | Path) -> Path:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scan_result_to_dict(result), indent=2))
    return path


def build_pole_scan_map_figure(
    scan_results_file: str | Path,
    best_result_file: str | Path | None = None,
    coordinate_mode: str = "initial",
) -> list[plt.Figure]: # type: ignore
    """Build separate static pole solution map figures from a pole-scan CSV."""
    if coordinate_mode not in {"initial", "fitted"}:
        raise ValueError("coordinate_mode must be 'initial' or 'fitted'.")

    x_column = "initial_lambda" if coordinate_mode == "initial" else "fitted_lambda"
    y_column = "initial_beta" if coordinate_mode == "initial" else "fitted_beta"
    coordinate_label = (
        "initial pole coordinates"
        if coordinate_mode == "initial"
        else "fitted pole coordinates"
    )

    columns = [
        "initial_lambda",
        "initial_beta",
        "fitted_lambda",
        "fitted_beta",
        "status",
        "chi_square",
        "fitted_period",
        "shadow_percent",
    ]
    data = pd.read_csv(scan_results_file, usecols=lambda column: column in columns)
    for column in columns:
        if column not in data.columns:
            data[column] = np.nan
    data = data[data["status"] == "success"].copy()
    for column in [
        "initial_lambda",
        "initial_beta",
        "fitted_lambda",
        "fitted_beta",
        "chi_square",
        "fitted_period",
        "shadow_percent",
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    best = None
    best_path = Path(best_result_file) if best_result_file is not None else None
    if best_path is not None and best_path.exists():
        best = json.loads(best_path.read_text())

    figures = []
    for column, label, _suffix in _POLE_SCAN_MAP_METRICS:
        fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        metric_data = data.dropna(subset=[x_column, y_column, column])
        x = metric_data[x_column].to_numpy()
        y = metric_data[y_column].to_numpy()
        z = metric_data[column].to_numpy()
        artist = None
        if len(metric_data) >= 3:
            try:
                triangulation = mtri.Triangulation(x, y)
                artist = axis.tricontourf(
                    triangulation, z, levels=100, cmap="coolwarm", zorder=-1
                )
            except (RuntimeError, ValueError):
                artist = None
        if artist is None:
            artist = axis.scatter(
                x,
                y,
                c=z,
                cmap="coolwarm",
                s=40,
                edgecolors="none",
                zorder=1,
            )
        fig.colorbar(artist, ax=axis, orientation="vertical", label=label)
        if (
            best is not None
            and best.get(x_column) is not None
            and best.get(y_column) is not None
        ):
            axis.scatter(
                [best[x_column]],
                [best[y_column]],
                marker="*",
                s=180,
                c="black",
                edgecolors="white",
                linewidths=0.8,
                label="Best",
            )
            axis.legend(loc="upper right")
        axis.set_xlim(0, 360)
        axis.set_ylim(-90, 90)
        axis.set_xlabel(r"$\lambda$ [deg]")
        axis.set_ylabel(r"$\beta$ [deg]")
        axis.set_title(label)
        axis.grid(alpha=0.25)

        if column == "chi_square" and best is not None:
            best_lambda = best.get(x_column)
            best_beta = best.get(y_column)
            if best_lambda is None:
                best_lambda = float("nan")
            if best_beta is None:
                best_beta = float("nan")
            axis.text(
                0.05,
                0.88,
                "Minimum chi-squared "
                f"{best.get('chi_square', float('nan')):.6g} w/\n"
                f"  ({coordinate_mode} lam, beta) = ({best_lambda:.1f}, "
                f"{best_beta:.1f}), "
                f"darkfacet {best.get('shadow_percent', float('nan')):.2f}%",
                size=10,
                transform=axis.transAxes,
            )
        fig.suptitle(f"Pole Grid Scan Map ({coordinate_label})")
        figures.append(fig)
    return figures


def _pole_scan_map_output_paths(output_file: str | Path) -> list[Path]:
    path = Path(output_file)
    return [
        path if suffix == "" else path.with_name(f"{path.stem}_{suffix}{path.suffix}")
        for _column, _label, suffix in _POLE_SCAN_MAP_METRICS
    ]


def save_pole_scan_map_matplotlib(
    scan_results_file: str | Path,
    best_result_file: str | Path | None,
    output_file: str | Path,
    coordinate_mode: str = "initial",
) -> list[Path]:
    """Save separate static pole solution maps similar to plot_polesolution.py."""
    paths = _pole_scan_map_output_paths(output_file)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    figures = build_pole_scan_map_figure(
        scan_results_file, best_result_file, coordinate_mode=coordinate_mode
    )
    for fig, path in zip(figures, paths):
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return paths
