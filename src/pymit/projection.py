from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection


@dataclass(frozen=True)
class SkyProjection:
    rotated_vertices: np.ndarray
    projected_vertices: np.ndarray
    visible_faces: list[int]
    hidden_faces: list[int]
    phase_degrees: float
    view_vector: np.ndarray
    sky_x_axis: np.ndarray
    sky_y_axis: np.ndarray

    @property
    def face_count(self) -> int:
        return len(self.visible_faces) + len(self.hidden_faces)


def _unit_vector(vector) -> np.ndarray:
    arr = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Vector must be non-zero.")
    return arr / norm


def _rotation_matrix_z(phase_degrees: float) -> np.ndarray:
    angle = np.deg2rad(phase_degrees)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array(
        [
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _phase_from_jd(
    jd: Optional[float],
    period_hours: Optional[float],
    zero_time: float,
    initial_rotation_angle: float,
    phase_degrees: Optional[float],
) -> float:
    if phase_degrees is not None:
        return float(phase_degrees) % 360.0
    if jd is None or period_hours is None:
        return float(initial_rotation_angle) % 360.0
    period_days = period_hours / 24.0
    if period_days == 0:
        raise ValueError("period_hours must be non-zero.")
    return (initial_rotation_angle + 360.0 * ((jd - zero_time) / period_days)) % 360.0


def first_observation_jd(lightcurve_source) -> float:
    """Return the first observation JD from a native lightcurve file or DataFrame."""
    if isinstance(lightcurve_source, pd.DataFrame):
        if "jd" not in lightcurve_source.columns or lightcurve_source.empty:
            raise ValueError("DataFrame lightcurve input must contain a non-empty jd column.")
        return float(lightcurve_source.iloc[0]["jd"])

    path = Path(lightcurve_source)
    with open(path, "r") as handle:
        first = handle.readline().strip()
        try:
            n_curves = int(first)
        except ValueError as e:
            raise ValueError(f"Invalid native lightcurve header in {path}: {first}") from e
        if n_curves <= 0:
            raise ValueError(f"Native lightcurve file has no curves: {path}")

        header = handle.readline().split()
        if not header:
            raise ValueError(f"Missing first curve header in {path}")
        n_points = int(header[0])
        if n_points <= 0:
            raise ValueError(f"First lightcurve has no points in {path}")

        first_point = handle.readline().split()
        if not first_point:
            raise ValueError(f"Missing first observation row in {path}")
        return float(first_point[0])


def _sky_basis(view_vector) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    view = _unit_vector(view_vector)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(view, reference)) > 0.95:
        reference = np.array([0.0, 1.0, 0.0])
    sky_x = _unit_vector(np.cross(reference, view))
    sky_y = _unit_vector(np.cross(view, sky_x))
    return view, sky_x, sky_y


def _face_vertices(vertices: np.ndarray, face: list[int]) -> np.ndarray:
    return vertices[[idx - 1 for idx in face]]


def _face_normal(face_vertices: np.ndarray) -> np.ndarray:
    if len(face_vertices) < 3:
        return np.zeros(3)
    normal = np.cross(face_vertices[1] - face_vertices[0], face_vertices[2] - face_vertices[0])
    norm = np.linalg.norm(normal)
    if norm == 0:
        return np.zeros(3)
    return normal / norm


def _face_area(face_vertices: np.ndarray) -> float:
    if len(face_vertices) < 3:
        return 0.0
    origin = face_vertices[0]
    area = 0.0
    for idx in range(1, len(face_vertices) - 1):
        area += np.linalg.norm(
            np.cross(face_vertices[idx] - origin, face_vertices[idx + 1] - origin)
        ) / 2.0
    return float(area)


def project_shape_to_sky(
    vertices: np.ndarray,
    faces: list[list[int]],
    *,
    jd: Optional[float] = None,
    period_hours: Optional[float] = None,
    zero_time: float = 0.0,
    initial_rotation_angle: float = 0.0,
    phase_degrees: Optional[float] = None,
    view_vector=(0.0, 0.0, 1.0),
) -> SkyProjection:
    """Rotate a shape model and project it onto a 2D sky plane."""
    phase = _phase_from_jd(
        jd, period_hours, zero_time, initial_rotation_angle, phase_degrees
    )
    rotated_vertices = np.asarray(vertices, dtype=float) @ _rotation_matrix_z(phase).T
    view, sky_x, sky_y = _sky_basis(view_vector)
    projected_vertices = np.column_stack(
        (rotated_vertices @ sky_x, rotated_vertices @ sky_y)
    )

    visible_faces = []
    hidden_faces = []
    for face_index, face in enumerate(faces):
        normal = _face_normal(_face_vertices(rotated_vertices, face))
        if np.dot(normal, view) > 0:
            visible_faces.append(face_index)
        else:
            hidden_faces.append(face_index)

    projection = SkyProjection(
        rotated_vertices=rotated_vertices,
        projected_vertices=projected_vertices,
        visible_faces=visible_faces,
        hidden_faces=hidden_faces,
        phase_degrees=phase,
        view_vector=view,
        sky_x_axis=sky_x,
        sky_y_axis=sky_y,
    )
    return projection


def sky_projection_dataframe(projection: SkyProjection) -> pd.DataFrame:
    rows = []
    for idx, (sky, rotated) in enumerate(
        zip(projection.projected_vertices, projection.rotated_vertices), start=1
    ):
        rows.append(
            {
                "vertex_index": idx,
                "sky_x": sky[0],
                "sky_y": sky[1],
                "rotated_x": rotated[0],
                "rotated_y": rotated[1],
                "rotated_z": rotated[2],
                "phase_degrees": projection.phase_degrees,
            }
        )
    return pd.DataFrame(rows)


def save_sky_projection_csv(projection: SkyProjection, output_file: str) -> None:
    sky_projection_dataframe(projection).to_csv(output_file, index=False)


def _projected_face_polygon(projection: SkyProjection, face: list[int]) -> np.ndarray:
    return projection.projected_vertices[[idx - 1 for idx in face]]


def _visible_face_color(normal: np.ndarray, view: np.ndarray) -> tuple[float, float, float, float]:
    illumination = max(float(np.dot(normal, view)), 0.0)
    shade = 0.24 + 0.76 * illumination
    base = np.array([0.43, 0.53, 0.62])
    highlight = np.array([0.92, 0.95, 0.98])
    rgb = base * (1.0 - shade) + highlight * shade
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 0.98)


def _set_equal_sky_limits(ax, projected_vertices: np.ndarray) -> None:
    x = projected_vertices[:, 0]
    y = projected_vertices[:, 1]
    center_x = (float(x.min()) + float(x.max())) / 2.0
    center_y = (float(y.min()) + float(y.max())) / 2.0
    span = max(float(x.max() - x.min()), float(y.max() - y.min()), 1.0)
    radius = span * 0.58
    ax.set_xlim(center_x - radius, center_x + radius)
    ax.set_ylim(center_y - radius, center_y + radius)


def plot_sky_projection(
    vertices: np.ndarray,
    faces: list[list[int]],
    *,
    save_path: Optional[str] = None,
    show: bool = True,
    **projection_kwargs,
) -> SkyProjection:
    """Plot a rotated model projected onto the sky plane."""
    projection = project_shape_to_sky(vertices, faces, **projection_kwargs)

    hidden_polygons = []
    for face_index in projection.hidden_faces:
        face = faces[face_index]
        hidden_polygons.append(_projected_face_polygon(projection, face))

    visible_polygons = []
    visible_colors = []
    for face_index in projection.visible_faces:
        face = faces[face_index]
        visible_polygons.append(_projected_face_polygon(projection, face))
        normal = _face_normal(_face_vertices(projection.rotated_vertices, face))
        visible_colors.append(_visible_face_color(normal, projection.view_vector))

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0b1020")
    ax.set_facecolor("#0b1020")

    if hidden_polygons:
        collection = PolyCollection(
            hidden_polygons,
            facecolors="#1f2937",
            edgecolors="#334155",
            linewidths=0.35,
            alpha=0.30,
            zorder=1,
        )
        ax.add_collection(collection)

    if visible_polygons:
        collection = PolyCollection(
            visible_polygons,
            facecolors=visible_colors,
            edgecolors="#dbeafe",
            linewidths=0.45,
            alpha=0.98,
            zorder=2,
        )
        ax.add_collection(collection)

    ax.scatter(
        projection.projected_vertices[:, 0],
        projection.projected_vertices[:, 1],
        s=2,
        color="#f8fafc",
        alpha=0.10,
        zorder=3,
    )
    ax.set_aspect("equal", adjustable="box")
    _set_equal_sky_limits(ax, projection.projected_vertices)
    ax.grid(color="#334155", alpha=0.22, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(0.8)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.set_xlabel("Sky X")
    ax.set_ylabel("Sky Y")
    ax.xaxis.label.set_color("#cbd5e1")
    ax.yaxis.label.set_color("#cbd5e1")
    ax.set_title(
        f"Sky Projection - phase {projection.phase_degrees:.1f} deg",
        color="#e5e7eb",
        pad=14,
        fontsize=13,
    )

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)
    return projection


def compute_synthetic_lightcurve(
    vertices: np.ndarray,
    faces: list[list[int]],
    *,
    n_steps: int = 72,
    jd: Optional[float] = None,
    period_hours: Optional[float] = None,
    zero_time: float = 0.0,
    initial_rotation_angle: float = 0.0,
    phase_degrees: Optional[float] = None,
    sun_vector=(1.0, 0.0, 1.0),
    view_vector=(0.0, 0.0, 1.0),
    lambert_coefficient: float = 0.1,
) -> pd.DataFrame:
    """
    Compute a simple fixed-geometry brightness curve from visible illuminated facets.

    This approximates DAMIT's fixed-position light-curve panel. It does not model
    shadowing or the full DAMIT scattering law.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    sun = _unit_vector(sun_vector)
    view = _unit_vector(view_vector)
    start_phase_degrees = _phase_from_jd(
        jd, period_hours, zero_time, initial_rotation_angle, phase_degrees
    )
    phases = np.linspace(0.0, 1.0, n_steps, endpoint=False)
    phase_degrees_values = (start_phase_degrees + phases * 360.0) % 360.0
    brightness = []

    for phase_degrees_value in phase_degrees_values:
        rotation = _rotation_matrix_z(phase_degrees_value)
        rotated_vertices = np.asarray(vertices, dtype=float) @ rotation.T
        total = 0.0
        for face in faces:
            face_vertices = _face_vertices(rotated_vertices, face)
            normal = _face_normal(face_vertices)
            area = _face_area(face_vertices)
            mu = max(float(np.dot(normal, view)), 0.0)
            mu0 = max(float(np.dot(normal, sun)), 0.0)
            if mu > 0 and mu0 > 0:
                total += area * mu0 * ((1.0 - lambert_coefficient) + lambert_coefficient * mu)
        brightness.append(total)

    values = np.asarray(brightness, dtype=float)
    if values.max() > 0:
        values = values / values.max()
    return pd.DataFrame(
        {
            "phase": phases,
            "phase_degrees": phase_degrees_values,
            "brightness": values,
        }
    )


def save_synthetic_lightcurve_csv(lightcurve: pd.DataFrame, output_file: str) -> None:
    lightcurve.to_csv(output_file, index=False)


def plot_synthetic_lightcurve(
    lightcurve: pd.DataFrame,
    *,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lightcurve["phase"], lightcurve["brightness"], "-o", markersize=3)
    ax.set_xlabel("Rotation phase")
    ax.set_ylabel("Normalized Flux")
    ax.set_title("Synthetic Light Curve")
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
