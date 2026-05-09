from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from pymit.shape import _triangulate_faces

PARAMS_FILE = Path(__file__).resolve().parents[2] / ".pymit_last_params.json"


def load_saved_params() -> dict:
    try:
        return json.loads(PARAMS_FILE.read_text())
    except Exception:
        return {}


def save_params(params: dict) -> None:
    PARAMS_FILE.write_text(json.dumps(params, indent=2))


@dataclass(frozen=True)
class InversionSettings:
    initial_lambda: float
    initial_lambda_fixed: bool
    initial_beta: float
    initial_beta_fixed: bool
    initial_period: float
    initial_period_fixed: bool
    phase_func_a: float
    phase_func_a_fixed: bool
    phase_func_d: float
    phase_func_d_fixed: bool
    phase_func_k: float
    phase_func_k_fixed: bool
    phase_func_c: float
    phase_func_c_fixed: bool
    convexity_regularization: float
    spherical_harmonics_degree: int
    spherical_harmonics_order: int
    number_of_rows: int
    iteration_stop_condition: int


def _fixed_flag(value: bool) -> int:
    return 1 if value else 0


def current_julian_date(now: datetime | None = None) -> float:
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)

    return 2440587.5 + now.timestamp() / 86400.0


def build_inversion_options(settings: InversionSettings) -> dict:
    return {
        "initial_lambda": settings.initial_lambda,
        "initial_lambda_fixed": _fixed_flag(settings.initial_lambda_fixed),
        "initial_beta": settings.initial_beta,
        "initial_beta_fixed": _fixed_flag(settings.initial_beta_fixed),
        "initial_period": settings.initial_period,
        "initial_period_fixed": _fixed_flag(settings.initial_period_fixed),
        "phase_func_a": settings.phase_func_a,
        "phase_func_a_fixed": _fixed_flag(settings.phase_func_a_fixed),
        "phase_func_d": settings.phase_func_d,
        "phase_func_d_fixed": _fixed_flag(settings.phase_func_d_fixed),
        "phase_func_k": settings.phase_func_k,
        "phase_func_k_fixed": _fixed_flag(settings.phase_func_k_fixed),
        "phase_func_c": settings.phase_func_c,
        "phase_func_c_fixed": _fixed_flag(settings.phase_func_c_fixed),
        "convexity_regularization": settings.convexity_regularization,
        "spherical_harmonics_degree": settings.spherical_harmonics_degree,
        "spherical_harmonics_order": settings.spherical_harmonics_order,
        "number_of_rows": settings.number_of_rows,
        "iteration_stop_condition": settings.iteration_stop_condition,
    }


def build_conjgradinv_options(
    convexity_weight: float,
    number_of_rows: int,
    number_of_iterations: int,
) -> dict:
    return {
        "convexity_weight": convexity_weight,
        "number_of_rows": number_of_rows,
        "number_of_iterations": number_of_iterations,
    }


@dataclass(frozen=True)
class GeneratedOutput:
    label: str
    path: Path
    mime_type: str


def save_uploaded_lightcurve(uploaded_file, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / Path(uploaded_file.name).name
    destination.write_bytes(bytes(uploaded_file.getbuffer()))
    return destination


def collect_generated_outputs(output_dir: Path, asteroid_name: str) -> list[GeneratedOutput]:
    base_name = asteroid_name.replace(" ", "_")
    candidates = [
        GeneratedOutput("OBJ model", output_dir / f"{base_name}.obj", "text/plain"),
        GeneratedOutput("Interactive model HTML", output_dir / f"{base_name}_model.html", "text/html"),
        GeneratedOutput("Static model PNG", output_dir / f"{base_name}_model.png", "image/png"),
        GeneratedOutput("Lightcurve plot PNG", output_dir / f"{base_name}_lightcurves.png", "image/png"),
        GeneratedOutput("Sky projection PNG", output_dir / f"{base_name}_sky_projection.png", "image/png"),
        GeneratedOutput("Sky projection CSV", output_dir / f"{base_name}_sky_projection.csv", "text/csv"),
        GeneratedOutput("Synthetic lightcurve PNG", output_dir / f"{base_name}_synthetic_lightcurve.png", "image/png"),
        GeneratedOutput("Synthetic lightcurve CSV", output_dir / f"{base_name}_synthetic_lightcurve.csv", "text/csv"),
        GeneratedOutput("Areas and normals TXT", output_dir / f"{base_name}_areas.txt", "text/plain"),
        GeneratedOutput("Modeled lightcurve CSV", output_dir / f"{base_name}_lc_output.csv", "text/csv"),
        GeneratedOutput("Input convexinv TXT", output_dir / f"{base_name}_input_convexinv.txt", "text/plain"),
        GeneratedOutput("Input conjgradinv TXT", output_dir / "input_conjgradinv", "text/plain"),
    ]
    return [candidate for candidate in candidates if candidate.path.exists()]


def build_model_figure(vertices: np.ndarray, faces: list[list[int]]) -> go.Figure:
    triangles = _triangulate_faces(faces)
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=[tri[0] for tri in triangles],
                j=[tri[1] for tri in triangles],
                k=[tri[2] for tri in triangles],
                color="lightgray",
                opacity=1.0,
                lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2, roughness=0.5),
                lightposition=dict(x=100, y=100, z=100),
            )
        ]
    )
    fig.update_layout(
        title="Asteroid Shape Model",
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig
