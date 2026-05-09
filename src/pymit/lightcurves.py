import csv
import math
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd


def _phase_angle_degrees(sun_vector, observer_vector) -> float:
    sun = tuple(float(value) for value in sun_vector)
    observer = tuple(float(value) for value in observer_vector)
    sun_norm = math.sqrt(sum(value * value for value in sun))
    observer_norm = math.sqrt(sum(value * value for value in observer))
    if sun_norm == 0 or observer_norm == 0:
        raise ValueError("Sun and observer vectors must be non-zero.")

    cos_angle = sum(s * o for s, o in zip(sun, observer)) / (sun_norm * observer_norm)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def _lightcurve_x_value(
    parts: list[str],
    t0: float,
    period_hours: Optional[float],
    zero_time: Optional[float],
) -> tuple[float, str]:
    if len(parts) >= 8:
        return (
            _phase_angle_degrees(
                (parts[2], parts[3], parts[4]),
                (parts[5], parts[6], parts[7]),
            ),
            "Solar Phase Angle (deg)",
        )
    if period_hours is not None and zero_time is not None:
        period_days = period_hours / 24.0
        if period_days == 0:
            raise ValueError("period_hours must be non-zero.")
        return ((float(parts[0]) - zero_time) / period_days) % 1.0, "Rotation Phase"
    return float(parts[0]) - t0, "Time (Days from curve start)"


def plot_lightcurves(
    lightcurve: Union[str, pd.DataFrame],
    output_file: str,
    save_path: Optional[str] = None,
    show: bool = True,
    max_curves: int = 3,
    period_hours: Optional[float] = None,
    zero_time: Optional[float] = None,
) -> None:
    """Plot observed vs modeled light curves."""
    temp_input = None
    if isinstance(lightcurve, pd.DataFrame):
        temp_input = "temp_plot_lcs_input.txt"
        dataframe_to_lcs_format(lightcurve, temp_input)
        actual_input_file = temp_input
    elif isinstance(lightcurve, str) and lightcurve.lower().endswith(".csv"):
        temp_input = "temp_plot_lcs_input.txt"
        csv_to_lcs_format(lightcurve, temp_input)
        actual_input_file = temp_input
    else:
        actual_input_file = lightcurve

    try:
        with open(actual_input_file, "r") as f_in, open(output_file, "r") as f_out:
            n_curves = int(f_in.readline().strip())

            plt.figure(figsize=(10, 6))
            x_label = "Time (Days from curve start)"

            for i in range(n_curves):
                header = f_in.readline().split()
                n_pts = int(header[0])

                x = []
                y_obs = []
                y_mod = []
                t0 = None
                x_label = "Time (Days from curve start)"

                for _ in range(n_pts):
                    parts = f_in.readline().split()
                    if t0 is None:
                        t0 = float(parts[0])
                    x_value, x_label = _lightcurve_x_value(
                        parts, t0, period_hours, zero_time
                    )
                    x.append(x_value)
                    y_obs.append(float(parts[1]))
                    y_mod.append(float(f_out.readline().strip()))

                if i < max_curves:
                    plt.plot(x, y_obs, "o", label=f"Observed Curve {i+1}")
                    plt.plot(x, y_mod, "x", label=f"Modeled Curve {i+1}")

            plt.xlabel(x_label)
            plt.ylabel("Brightness")
            plt.title("Observed vs Modeled Brightness vs Phase")
            plt.legend()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")

            if show:
                plt.show()

            plt.close()
    finally:
        if temp_input and Path(temp_input).exists():
            Path(temp_input).unlink()


def dataframe_to_lcs_format(df: pd.DataFrame, output_file: str) -> None:
    """Convert a lightcurve DataFrame into the text format expected by convexinv."""
    curves = {}

    required_cols = [
        "jd",
        "brightness",
        "sun_x",
        "sun_y",
        "sun_z",
        "earth_x",
        "earth_y",
        "earth_z",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in DataFrame: {col}")

    for row_idx, row in df.iterrows():
        try:
            jd = float(row["jd"])
            bright = float(row["brightness"])
            sx, sy, sz = float(row["sun_x"]), float(row["sun_y"]), float(row["sun_z"])
            ex, ey, ez = (
                float(row["earth_x"]),
                float(row["earth_y"]),
                float(row["earth_z"]),
            )
        except ValueError as e:
            raise ValueError(f"Invalid data format in DataFrame row {row_idx}: {e}")

        cid = str(row.get("curve_id", "1"))
        is_rel = int(row.get("is_relative", 0))

        if cid not in curves:
            curves[cid] = {"is_relative": is_rel, "points": []}

        curves[cid]["points"].append((jd, bright, sx, sy, sz, ex, ey, ez))

    _write_lcs_dict_to_file(curves, output_file)


def csv_to_lcs_format(csv_file: str, output_file: str) -> None:
    """Convert a CSV file into the text format expected by convexinv."""
    curves = {}
    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            try:
                jd = float(row["jd"])
                bright = float(row["brightness"])
                sx, sy, sz = (
                    float(row["sun_x"]),
                    float(row["sun_y"]),
                    float(row["sun_z"]),
                )
                ex, ey, ez = (
                    float(row["earth_x"]),
                    float(row["earth_y"]),
                    float(row["earth_z"]),
                )
            except KeyError as e:
                raise ValueError(f"Missing required column in CSV: {e}")
            except ValueError as e:
                raise ValueError(f"Invalid data format in CSV row {row_idx}: {e}")

            cid = row.get("curve_id", "1")
            is_rel = int(row.get("is_relative", "0"))

            if cid not in curves:
                curves[cid] = {"is_relative": is_rel, "points": []}

            curves[cid]["points"].append((jd, bright, sx, sy, sz, ex, ey, ez))

    _write_lcs_dict_to_file(curves, output_file)


def _write_lcs_dict_to_file(curves: dict, output_file: str) -> None:
    with open(output_file, "w") as f_out:
        f_out.write(f"{len(curves)}\n")

        for cid, data in curves.items():
            pts = data["points"]
            out_is_rel = 0 if data["is_relative"] == 1 else 1
            f_out.write(f"{len(pts)} {out_is_rel}\n")

            for pt in pts:
                jd, br, sx, sy, sz, ex, ey, ez = pt
                f_out.write(
                    f"{jd:.6f} {br:.6e}  {sx:.6e} {sy:.6e} {sz:.6e}  {ex:.6e} {ey:.6e} {ez:.6e}\n"
                )
