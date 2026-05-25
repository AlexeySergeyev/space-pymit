import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PeriodScanResult:
    period_hours: float
    rms: float
    chi_square: float
    iterations: int
    shadow_percent: float


_PERIOD_SCAN_FIELDS = [
    "period_hours",
    "rms",
    "chi_square",
    "iterations",
    "shadow_percent",
]


def parse_period_scan_output(output_file: str | Path) -> list[PeriodScanResult]:
    """Parse DAMIT period_scan output rows."""
    results = []
    for line in Path(output_file).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            continue
        try:
            result = PeriodScanResult(
                period_hours=float(parts[0]),
                rms=float(parts[1]),
                chi_square=float(parts[2]),
                iterations=int(parts[3]),
                shadow_percent=float(parts[4]),
            )
        except ValueError:
            continue
        results.append(result)
    return results


def find_best_period_scan_result(
    results: Iterable[PeriodScanResult],
) -> PeriodScanResult:
    """Return the period scan row with the lowest chi-square."""
    result_list = list(results)
    if not result_list:
        raise ValueError("period_scan did not produce any parseable result rows.")
    return min(result_list, key=lambda result: result.chi_square)


def write_period_scan_csv(
    results: Iterable[PeriodScanResult], output_file: str | Path
) -> Path:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_PERIOD_SCAN_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    return path


def save_period_scan_plot(
    results: Iterable[PeriodScanResult],
    output_file: str | Path,
    best_result: Optional[PeriodScanResult] = None,
) -> Path:
    result_list = list(results)
    if not result_list:
        raise ValueError("Cannot plot an empty period_scan result set.")

    if best_result is None:
        best_result = find_best_period_scan_result(result_list)

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    periods = [result.period_hours for result in result_list]
    chi_squares = [result.chi_square for result in result_list]
    rms_values = [result.rms for result in result_list]

    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.plot(periods, chi_squares, color="#1f77b4", linewidth=1.5, label="chi-square")
    axis.scatter(periods, chi_squares, color="#1f77b4", s=18)
    axis.scatter(
        [best_result.period_hours],
        [best_result.chi_square],
        marker="*",
        s=180,
        c="black",
        edgecolors="white",
        linewidths=0.8,
        label="Best period",
        zorder=3,
    )
    axis.set_xlabel("Period [h]")
    axis.set_ylabel(r"$\chi^2$")
    axis.grid(alpha=0.25)

    rms_axis = axis.twinx()
    rms_axis.plot(periods, rms_values, color="#d62728", linewidth=1.0, alpha=0.75, label="rms")
    rms_axis.set_ylabel("RMS")

    lines, labels = axis.get_legend_handles_labels()
    rms_lines, rms_labels = rms_axis.get_legend_handles_labels()
    axis.legend(lines + rms_lines, labels + rms_labels, loc="best")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
