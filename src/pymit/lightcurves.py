import csv
import math
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import plotly.graph_objects as go


REQUIRED_LIGHTCURVE_COLUMNS = [
    "jd",
    "brightness",
    "sun_x",
    "sun_y",
    "sun_z",
    "earth_x",
    "earth_y",
    "earth_z",
]

_PLOTLY_MARKER_COLORS = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


def validate_lightcurve_dataframe_columns(
    df: pd.DataFrame, source_label: str = "DataFrame"
) -> None:
    """Validate that tabular lightcurve data can be converted to convexinv input."""
    missing = [column for column in REQUIRED_LIGHTCURVE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"{source_label} lightcurve input is missing required columns: "
            f"{', '.join(missing)}. Use a native DAMIT/convexinv .txt/.lc file, "
            "or provide a CSV with jd, brightness, sun_x, sun_y, sun_z, "
            "earth_x, earth_y, and earth_z columns."
        )


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


def build_lightcurve_figure(
    lightcurve: Union[str, pd.DataFrame],
    output_file: str,
    max_curves: int = 3,
    period_hours: Optional[float] = None,
    zero_time: Optional[float] = None,
) -> go.Figure:
    """Build an interactive observed-vs-modeled lightcurve figure."""
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
            x_label = "Time (Days from curve start)"
            fig = go.Figure()

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
                    curve_number = i + 1
                    color = _PLOTLY_MARKER_COLORS[i % len(_PLOTLY_MARKER_COLORS)]
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y_obs,
                            mode="markers",
                            name=f"Observed Curve {curve_number}",
                            marker=dict(symbol="circle", size=8, color=color),
                            customdata=[
                                [curve_number, "Observed"] for _ in x
                            ],
                            hovertemplate=(
                                "Curve %{customdata[0]}<br>"
                                "Type: %{customdata[1]}<br>"
                                f"{x_label}: %{{x:.6g}}<br>"
                                "Brightness: %{y:.6g}<extra></extra>"
                            ),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y_mod,
                            mode="markers",
                            name=f"Modeled Curve {curve_number}",
                            marker=dict(symbol="x", size=9, color=color),
                            customdata=[
                                [curve_number, "Modeled"] for _ in x
                            ],
                            hovertemplate=(
                                "Curve %{customdata[0]}<br>"
                                "Type: %{customdata[1]}<br>"
                                f"{x_label}: %{{x:.6g}}<br>"
                                "Brightness: %{y:.6g}<extra></extra>"
                            ),
                        )
                    )

            fig.update_layout(
                title="Observed vs Modeled Brightness vs Phase",
                xaxis_title=x_label,
                yaxis_title="Brightness",
                template="plotly_white",
                legend_title_text="Lightcurve",
                margin=dict(l=50, r=20, t=60, b=50),
            )
            return fig
    finally:
        if temp_input and Path(temp_input).exists():
            Path(temp_input).unlink()


def plot_lightcurves(
    lightcurve: Union[str, pd.DataFrame],
    output_file: str,
    save_path: Optional[str] = None,
    show: bool = True,
    max_curves: int = 3,
    period_hours: Optional[float] = None,
    zero_time: Optional[float] = None,
) -> go.Figure:
    """Plot observed vs modeled light curves as an interactive Plotly figure."""
    fig = build_lightcurve_figure(
        lightcurve,
        output_file,
        max_curves=max_curves,
        period_hours=period_hours,
        zero_time=zero_time,
    )
    if save_path:
        fig.write_html(save_path)
    if show:
        fig.show()
    return fig


def dataframe_to_lcs_format(df: pd.DataFrame, output_file: str) -> None:
    """Convert a lightcurve DataFrame into the text format expected by convexinv."""
    curves = {}
    validate_lightcurve_dataframe_columns(df)

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


def normalize_native_lcs_format(input_file: Union[str, Path], output_file: Union[str, Path]) -> bool:
    """
    Rewrite a native DAMIT/convexinv lightcurve file with internally consistent
    curve counts.

    Some lcs4DAMIT exports contain valid point rows but an overstated block
    count in a curve header. convexinv reads exactly the declared number of
    rows, so this function repairs those headers while preserving the numeric
    values and writing canonical DAMIT-style rows. Returns True when the output
    differs from the input.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    original_text = input_path.read_text()
    lines = [line.strip() for line in original_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Native lightcurve file is empty: {input_path}")

    first = lines[0].split()
    if len(first) != 1:
        raise ValueError(f"Invalid native lightcurve curve-count header in {input_path}: {lines[0]}")
    try:
        declared_curves = int(first[0])
    except ValueError as exc:
        raise ValueError(f"Invalid native lightcurve curve-count header in {input_path}: {lines[0]}") from exc

    index = 1
    curves = []
    changed = False
    for curve_index in range(declared_curves):
        if index >= len(lines):
            changed = True
            break

        header = lines[index].split()
        if len(header) < 2:
            raise ValueError(
                f"Invalid native lightcurve block header at line {index + 1} in {input_path}: {lines[index]}"
            )
        try:
            declared_points = int(header[0])
            relative_flag = int(header[1])
        except ValueError as exc:
            raise ValueError(
                f"Invalid native lightcurve block header at line {index + 1} in {input_path}: {lines[index]}"
            ) from exc
        index += 1

        points = []
        while index < len(lines) and len(points) < declared_points:
            parts = lines[index].split()
            remaining_curves = declared_curves - curve_index - 1
            if remaining_curves > 0 and _looks_like_native_block_header(parts):
                changed = True
                break
            if len(parts) < 8:
                raise ValueError(
                    f"Invalid native lightcurve data row at line {index + 1} in {input_path}: {lines[index]}"
                )
            try:
                point = tuple(float(value) for value in parts[:8])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid native lightcurve data row at line {index + 1} in {input_path}: {lines[index]}"
                ) from exc
            points.append(point)
            index += 1

        if len(points) != declared_points:
            changed = True
        curves.append({"is_relative": relative_flag, "points": points})

    if index < len(lines):
        if not curves:
            raise ValueError(
                f"Native lightcurve file has extra rows but no curves in {input_path}."
            )
        while index < len(lines):
            parts = lines[index].split()
            if len(parts) < 8:
                raise ValueError(
                    f"Invalid native lightcurve data row at line {index + 1} in {input_path}: {lines[index]}"
                )
            try:
                point = tuple(float(value) for value in parts[:8])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid native lightcurve data row at line {index + 1} in {input_path}: {lines[index]}"
                ) from exc
            curves[-1]["points"].append(point)
            index += 1
        changed = True

    output_lines = [str(len(curves))]
    for curve in curves:
        output_lines.append(f"{len(curve['points'])} {curve['is_relative']}")
        for point in curve["points"]:
            output_lines.append(_format_native_lightcurve_row(point))
    output_text = "\n".join(output_lines) + "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text)

    if len(curves) != declared_curves:
        changed = True
    return changed or output_text != original_text


def _format_native_lightcurve_row(point: tuple[float, ...]) -> str:
    jd, brightness, sx, sy, sz, ex, ey, ez = point
    return (
        f"{jd:.6f} {brightness:.6e} "
        f"{sx:.6e} {sy:.6e} {sz:.6e} "
        f"{ex:.6e} {ey:.6e} {ez:.6e}"
    )


def _looks_like_native_block_header(parts: list[str]) -> bool:
    if len(parts) != 2:
        return False
    try:
        int(parts[0])
        int(parts[1])
    except ValueError:
        return False
    return True


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
