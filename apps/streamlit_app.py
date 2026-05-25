from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import contextlib
import io
import sys
import threading

import pandas as pd

# python -m streamlit run apps/streamlit_app.py

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import streamlit as st

from pymit import AsteroidModeler
from pymit.gui_helpers import (
    InversionSettings,
    build_conjgradinv_options,
    build_inversion_options,
    build_model_figure,
    collect_generated_outputs,
    current_julian_date,
    load_saved_params,
    save_params,
    save_uploaded_lightcurve,
)
from pymit.pole_scan import build_pole_scan_map_figure


DEFAULT_DAMIT_URL = (
    "https://damit.cuni.cz/projects/damit/LightCurves/"
    "exportAllForAsteroid/7753/plaintext/A7753.lc.txt"
)

_PARAM_DEFAULTS = {
    "asteroid_name": "A7753",
    "output_dir": "gui_output/A7753",
    "input_mode": "DAMIT URL",
    "damit_url": DEFAULT_DAMIT_URL,
    "initial_lambda": 269.0,
    "initial_lambda_fixed": True,
    "initial_beta": 62.0,
    "initial_beta_fixed": True,
    "initial_period": 33.0,
    "initial_period_fixed": True,
    "phase_func_a": 0.5,
    "phase_func_a_fixed": False,
    "phase_func_d": 0.1,
    "phase_func_d_fixed": False,
    "phase_func_k": -1.05,
    "phase_func_k_fixed": False,
    "phase_func_c": 0.1,
    "convexity_regularization": 0.1,
    "spherical_harmonics_degree": 6,
    "spherical_harmonics_order": 6,
    "number_of_rows": 8,
    "iteration_stop_condition": 50,
    "run_convex_inversion": True,
    "run_period_scan": False,
    "period_scan_start": 32.0,
    "period_scan_end": 34.0,
    "period_scan_interval_coefficient": 0.8,
    "period_scan_convexity_weight": 0.1,
    "period_scan_spherical_harmonics_degree": 3,
    "period_scan_spherical_harmonics_order": 3,
    "period_scan_number_of_rows": 4,
    "period_scan_minimum_iterations": 10,
    "period_scan_workers": 1,
    "convexity_weight": 0.2,
    "reconstruction_iterations": 100,
    "run_minkowski_reconstruction": True,
    "generate_projection_products": True,
    "synthetic_lightcurve_steps": 72,
    "run_pole_grid_scan": False,
    "pole_grid_n": 3,
    "pole_grid_workers": 1,
    "pole_scan_map_by_fitted": False,
}

_PERSISTENT_KEYS = list(_PARAM_DEFAULTS.keys()) + ["projection_jd"]


def _format_pipeline_failure_message(error: str) -> str:
    if error.startswith("Pipeline execution failed:"):
        return error
    return f"Pipeline execution failed: {error}"


@dataclass
class PipelineJobState:
    status: str = "running"
    log: str = ""
    result: dict | None = None
    error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append_log(self, text: str) -> None:
        with self._lock:
            self.log += text

    def mark_succeeded(self, result: dict) -> None:
        with self._lock:
            self.status = "succeeded"
            self.result = result
            self.error = None
            self.finished_at = datetime.now(timezone.utc)

    def mark_failed(self, exc: Exception) -> None:
        with self._lock:
            error_message = _format_pipeline_failure_message(str(exc))
            self.status = "failed"
            self.result = None
            self.error = error_message
            if self.log and not self.log.endswith("\n"):
                self.log += "\n"
            self.log += f"{error_message}\n"
            self.finished_at = datetime.now(timezone.utc)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "log": self.log,
                "result": self.result,
                "error": self.error,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }


class JobLogBuffer(io.StringIO):
    def __init__(self, job_state: PipelineJobState):
        super().__init__()
        self.job_state = job_state

    def write(self, text: str) -> int:
        written = super().write(text)
        self.job_state.append_log(text)
        return written


def _log_text_area_key(key_prefix: str, value: str) -> str:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest()
    return f"{key_prefix}_{digest}"


def _render_log_text_area(
    label: str,
    value: str,
    key_prefix: str,
    disabled: bool = True,
    height: int = 280,
) -> None:
    st.text_area(
        label,
        value=value,
        height=height,
        key=_log_text_area_key(key_prefix, value),
        disabled=disabled,
    )


def _init_session_defaults() -> None:
    """Seed st.session_state from saved JSON (or hardcoded defaults) once per load."""
    saved = load_saved_params()
    for key, default in _PARAM_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = saved.get(key, default)
    if "projection_jd" not in st.session_state:
        st.session_state["projection_jd"] = saved.get("projection_jd", current_julian_date())


def _save_current_params() -> None:
    """Persist all sidebar widget values to disk."""
    save_params({key: st.session_state[key] for key in _PERSISTENT_KEYS if key in st.session_state})


def render_sidebar():
    with st.sidebar.expander("Input", expanded=True):
        asteroid_name = st.text_input("Asteroid name", key="asteroid_name")
        output_dir = st.text_input("Output directory", key="output_dir")
        input_mode = st.radio(
            "Lightcurve input", ["DAMIT URL", "Upload file"], horizontal=True, key="input_mode"
        )

        damit_url = ""
        uploaded_file = None
        if input_mode == "DAMIT URL":
            damit_url = st.text_input("DAMIT plaintext URL", key="damit_url")
        else:
            uploaded_file = st.file_uploader(
                "CSV or native DAMIT text", type=["csv", "txt", "lc"]
            )

    run_period_scan = st.sidebar.checkbox("Period scan", key="run_period_scan")
    with st.sidebar.expander("Period scan", expanded=False):
        period_scan_start = st.number_input(
            "Period start [hours]", format="%.6f", key="period_scan_start"
        )
        period_scan_end = st.number_input(
            "Period end [hours]", format="%.6f", key="period_scan_end"
        )
        period_scan_interval_coefficient = st.number_input(
            "Period interval coefficient",
            min_value=0.000001,
            format="%.6f",
            key="period_scan_interval_coefficient",
        )
        period_scan_convexity_weight = st.number_input(
            "Period scan convexity weight",
            format="%.4f",
            key="period_scan_convexity_weight",
        )
        period_scan_spherical_harmonics_degree = st.number_input(
            "Period scan spherical harmonics degree",
            min_value=1,
            step=1,
            key="period_scan_spherical_harmonics_degree",
        )
        period_scan_spherical_harmonics_order = st.number_input(
            "Period scan spherical harmonics order",
            min_value=1,
            step=1,
            key="period_scan_spherical_harmonics_order",
        )
        period_scan_number_of_rows = st.number_input(
            "Period scan rows",
            min_value=1,
            step=1,
            key="period_scan_number_of_rows",
        )
        period_scan_minimum_iterations = st.number_input(
            "Period scan minimum iterations",
            min_value=0,
            step=1,
            key="period_scan_minimum_iterations",
        )
        period_scan_workers = st.number_input(
            "Period scan workers",
            min_value=1,
            step=1,
            key="period_scan_workers",
        )

    run_convex_inversion = st.sidebar.checkbox(
        "Convex inversion",
        key="run_convex_inversion",
    )
    with st.sidebar.expander("Convex inversion", expanded=False):
        settings = InversionSettings(
            initial_lambda=st.number_input("Initial lambda [deg]", key="initial_lambda"),
            initial_lambda_fixed=st.checkbox("Lambda free", key="initial_lambda_fixed"),
            initial_beta=st.number_input("Initial beta [deg]", key="initial_beta"),
            initial_beta_fixed=st.checkbox("Beta free", key="initial_beta_fixed"),
            initial_period=st.number_input(
                "Initial period [hours]", format="%.6f", key="initial_period"
            ),
            initial_period_fixed=st.checkbox("Period free", key="initial_period_fixed"),
            phase_func_a=st.number_input(
                "LSL p2 phase amplitude a",
                format="%.4f",
                help="DAMIT p2 for the LSL scattering model. Phase function amplitude a.",
                key="phase_func_a",
            ),
            phase_func_a_fixed=st.checkbox(
                "Fit p2",
                help="Allow convexinv to fit DAMIT p2/a. This can cause singular matrix errors unless the lightcurves constrain phase-angle behavior.",
                key="phase_func_a_fixed",
            ),
            phase_func_d=st.number_input(
                "LSL p3 phase width d",
                min_value=0.0,
                format="%.4f",
                help="DAMIT p3 for the LSL scattering model. Phase function width d.",
                key="phase_func_d",
            ),
            phase_func_d_fixed=st.checkbox(
                "Fit p3",
                help="Allow convexinv to fit DAMIT p3/d. This can cause singular matrix errors unless the lightcurves constrain phase-angle behavior.",
                key="phase_func_d_fixed",
            ),
            phase_func_k=st.number_input(
                "LSL p4 phase slope k",
                format="%.4f",
                help="DAMIT p4 for the LSL scattering model. Phase function linear slope k.",
                key="phase_func_k",
            ),
            phase_func_k_fixed=st.checkbox(
                "Fit p4",
                help="Allow convexinv to fit DAMIT p4/k. This can cause singular matrix errors unless the lightcurves constrain phase-angle behavior.",
                key="phase_func_k_fixed",
            ),
            phase_func_c=st.number_input(
                "LSL p1 Lambert coefficient c",
                min_value=0.0,
                max_value=1.0,
                format="%.4f",
                help="DAMIT p1 for the LSL scattering model. Lambertian fraction c.",
                key="phase_func_c",
            ),
            phase_func_c_fixed=False,
            convexity_regularization=st.number_input(
                "Convexity regularization", format="%.4f", key="convexity_regularization"
            ),
            spherical_harmonics_degree=st.number_input(
                "Spherical harmonics degree", min_value=1, step=1, key="spherical_harmonics_degree"
            ),
            spherical_harmonics_order=st.number_input(
                "Spherical harmonics order", min_value=1, step=1, key="spherical_harmonics_order"
            ),
            number_of_rows=st.number_input(
                "Triangulation rows", min_value=1, step=1, key="number_of_rows"
            ),
            iteration_stop_condition=st.number_input(
                "Iteration stop condition", min_value=1, step=1, key="iteration_stop_condition"
            ),
        )

    run_minkowski_reconstruction = st.sidebar.checkbox(
        "Minkowski reconstruction",
        key="run_minkowski_reconstruction",
    )
    with st.sidebar.expander("Minkowski reconstruction", expanded=False):
        convexity_weight = st.number_input(
            "Convexity weight", format="%.4f", key="convexity_weight"
        )
        reconstruction_iterations = st.number_input(
            "Reconstruction iterations", min_value=1, step=1, key="reconstruction_iterations"
        )

    run_pole_grid_scan = st.sidebar.checkbox("Pole grid search", key="run_pole_grid_scan")
    with st.sidebar.expander("Pole grid scan", expanded=False):
        pole_grid_n = st.number_input(
            "Golden spiral N", min_value=0, step=1, key="pole_grid_n"
        )
        st.caption(f"Runs {2 * int(pole_grid_n) + 1} initial pole guesses.")
        pole_grid_workers = st.number_input(
            "Pole scan workers", min_value=1, step=1, key="pole_grid_workers"
        )
        pole_scan_map_by_fitted = st.checkbox(
            "Plot map by fitted pole values", key="pole_scan_map_by_fitted"
        )

    generate_projection_products = st.sidebar.checkbox(
        "Sky projection", key="generate_projection_products"
    )
    with st.sidebar.expander("Sky projection and synthetic lightcurve", expanded=False):
        projection_jd = st.number_input(
            "Julian Date", key="projection_jd", format="%.6f"
        )
        synthetic_lightcurve_steps = st.number_input(
            "Synthetic lightcurve samples", min_value=8, step=1, key="synthetic_lightcurve_steps"
        )

    submitted = st.sidebar.button("Run modeling", type="primary")

    return {
        "submitted": submitted,
        "asteroid_name": asteroid_name,
        "output_dir": Path(output_dir),
        "input_mode": input_mode,
        "damit_url": damit_url,
        "uploaded_file": uploaded_file,
        "run_convex_inversion": run_convex_inversion,
        "run_minkowski_reconstruction": run_minkowski_reconstruction,
        "generate_projection_products": generate_projection_products,
        "projection_jd": projection_jd,
        "synthetic_lightcurve_steps": synthetic_lightcurve_steps,
        "run_period_scan": run_period_scan,
        "period_scan_options": {
            "period_start": period_scan_start,
            "period_end": period_scan_end,
            "period_interval_coefficient": period_scan_interval_coefficient,
            "convexity_weight": period_scan_convexity_weight,
            "spherical_harmonics_degree": int(period_scan_spherical_harmonics_degree),
            "spherical_harmonics_order": int(period_scan_spherical_harmonics_order),
            "number_of_rows": int(period_scan_number_of_rows),
            "phase_func_a": settings.phase_func_a,
            "phase_func_a_fixed": 1 if settings.phase_func_a_fixed else 0,
            "phase_func_d": settings.phase_func_d,
            "phase_func_d_fixed": 1 if settings.phase_func_d_fixed else 0,
            "phase_func_k": settings.phase_func_k,
            "phase_func_k_fixed": 1 if settings.phase_func_k_fixed else 0,
            "phase_func_c": settings.phase_func_c,
            "phase_func_c_fixed": 0,
            "iteration_stop_condition": settings.iteration_stop_condition,
            "minimum_number_of_iterations": int(period_scan_minimum_iterations),
        },
        "period_scan_workers": int(period_scan_workers),
        "run_pole_grid_scan": run_pole_grid_scan,
        "pole_grid_n": int(pole_grid_n),
        "pole_grid_workers": int(pole_grid_workers),
        "pole_scan_map_coordinate_mode": (
            "fitted" if pole_scan_map_by_fitted else "initial"
        ),
        "inversion_options": build_inversion_options(settings),
        "conjgradinv_options": build_conjgradinv_options(
            convexity_weight=convexity_weight,
            number_of_rows=settings.number_of_rows,
            number_of_iterations=reconstruction_iterations,
        ),
    }


def resolve_lightcurve_source(config: dict) -> str:
    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if config["input_mode"] == "DAMIT URL":
        if not config["damit_url"].strip():
            raise ValueError("DAMIT URL is required.")
        return config["damit_url"].strip()

    if config["uploaded_file"] is None:
        raise ValueError("Upload a CSV or native DAMIT text file.")
    return str(save_uploaded_lightcurve(config["uploaded_file"], output_dir))


def _lightcurve_file_for_static_plot(
    modeler: AsteroidModeler, output_dir: Path, base_name: str
) -> Path:
    if modeler.lightcurve_file is not None:
        return Path(modeler.lightcurve_file)
    if modeler.lightcurves is None:
        raise ValueError("No lightcurve data available for folded residual plotting.")

    from pymit.lightcurves import dataframe_to_lcs_format

    lightcurve_file = output_dir / f"{base_name}_lcs.txt"
    dataframe_to_lcs_format(modeler.lightcurves, str(lightcurve_file))
    return lightcurve_file


def _zero_time_for_static_plot(output_dir: Path, base_name: str, lightcurve_file: Path) -> float:
    from utils.plot_lightcurves import (
        _parse_zero_time_from_param_file,
        first_observation_jd,
    )

    param_file = output_dir / f"{base_name}_input_convexinv.txt"
    if param_file.exists():
        return _parse_zero_time_from_param_file(param_file)
    return first_observation_jd(lightcurve_file)


def generate_folded_residual_plot(
    modeler: AsteroidModeler,
    output_dir: Path,
    asteroid_name: str,
    max_curves: int = 1000,
) -> Path | None:
    fit_result = modeler.fit_result or {}
    period_hours = fit_result.get("period")
    if period_hours is None:
        print(
            "Folded residual plot skipped: fitted model period is unavailable.",
            flush=True,
        )
        return None

    from utils.plot_lightcurves import read_lightcurve_records, plot_folded_residuals

    base_name = asteroid_name.replace(" ", "_")
    model_output_file = output_dir / f"{base_name}_lc_output.csv"
    if not model_output_file.exists():
        print(
            f"Folded residual plot skipped: modeled lightcurve file not found at {model_output_file}.",
            flush=True,
        )
        return None

    lightcurve_file = _lightcurve_file_for_static_plot(modeler, output_dir, base_name)
    zero_time = _zero_time_for_static_plot(output_dir, base_name, lightcurve_file)
    records = read_lightcurve_records(lightcurve_file, model_output_file)
    save_path = output_dir / f"{base_name}_lightcurves_folded_residuals.png"
    plot_folded_residuals(
        records,
        float(period_hours),
        zero_time,
        save_path,
        max_curves=max_curves,
    )
    return save_path


def run_modeling_job(config: dict, source: str, log_stream: io.StringIO | None = None) -> dict:
    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    modeler = AsteroidModeler(
        asteroid_name=config["asteroid_name"],
        output_dir=str(output_dir),
    )

    active_log_stream = log_stream if log_stream is not None else io.StringIO()
    with contextlib.redirect_stdout(active_log_stream), contextlib.redirect_stderr(active_log_stream):
        vertices = None
        faces = None
        lightcurve_figure = None
        folded_residual_plot = None
        run_convex_inversion = config.get("run_convex_inversion", True)
        run_minkowski_reconstruction = config.get("run_minkowski_reconstruction", True)
        run_pole_grid_scan = config.get("run_pole_grid_scan", False)
        run_period_scan = config.get("run_period_scan", False)

        if not (
            run_period_scan
            or run_convex_inversion
            or run_pole_grid_scan
        ):
            raise ValueError("Select at least one modeling stage to run.")
        if run_minkowski_reconstruction and not (run_convex_inversion or run_pole_grid_scan):
            raise ValueError("Minkowski reconstruction requires convex inversion or pole grid scan.")

        modeler.load_lightcurves(source)
        modeler.load_parameters(
            inversion_json=config["inversion_options"],
            conjgradinv_json=config["conjgradinv_options"],
        )
        if run_period_scan:
            print("Running period scan...", flush=True)
            best_period = modeler.run_period_scan(
                period_scan_options=config["period_scan_options"],
                workers=config["period_scan_workers"],
                verbose=True,
            )
            if modeler.fit_result is None:
                modeler.fit_result = {
                    "initial_period_from_period_scan": best_period.period_hours,
                    "period_scan_chi_square": best_period.chi_square,
                }
        if run_pole_grid_scan:
            print("Running pole grid scan...", flush=True)
            vertices, faces = modeler.run_pole_grid_scan(
                n=config["pole_grid_n"],
                workers=config["pole_grid_workers"],
                verbose=True,
            )
        elif run_convex_inversion:
            print("Running convex inversion...", flush=True)
            modeler.run_convex_inversion(verbose=True)
            print("Convex inversion complete.", flush=True)

            if run_minkowski_reconstruction:
                print("Running Minkowski reconstruction...", flush=True)
                vertices, faces = modeler.run_minkowski_reconstruction(verbose=True)
                print("Minkowski reconstruction complete.", flush=True)

        if run_pole_grid_scan or run_convex_inversion:
            print("Plotting observed vs modeled lightcurves...", flush=True)
            lightcurve_figure = modeler.plot_lightcurves_results(
                save=True, show=False, max_curves=1000
            )
            print("Plotting folded residual lightcurve...", flush=True)
            folded_residual_plot = generate_folded_residual_plot(
                modeler,
                output_dir,
                config["asteroid_name"],
                max_curves=1000,
            )

        if vertices is not None and faces is not None:
            print("Plotting static 3D model...", flush=True)
            modeler.plot_model(save=True, show=False)
            print("Generating interactive 3D model...", flush=True)
            modeler.plot_model_plotly(save=True, show=False)
            print("Exporting OBJ model...", flush=True)
            modeler.export_obj()

        if (
            config["generate_projection_products"]
            and vertices is not None
            and faces is not None
        ):
            print("Generating sky projection...", flush=True)
            modeler.plot_sky_projection(
                save=True,
                show=False,
                jd=config["projection_jd"],
            )
            print("Generating synthetic lightcurve...", flush=True)
            modeler.plot_synthetic_lightcurve(
                save=True,
                show=False,
                n_steps=config["synthetic_lightcurve_steps"],
                jd=config["projection_jd"],
            )

        base_name = config["asteroid_name"].replace(" ", "_")
        print(
            f"Pipeline complete. All selected outputs for {base_name} preserved in '{str(output_dir)}/'.",
            flush=True,
        )
    active_log_stream.flush()

    return {
        "modeler": modeler,
        "vertices": vertices,
        "faces": faces,
        "lightcurve_figure": lightcurve_figure,
        "folded_residual_plot": folded_residual_plot,
        "fit_result": modeler.fit_result,
        "log": active_log_stream.getvalue(),
        "output_dir": output_dir,
    }


def start_modeling_job(config: dict) -> PipelineJobState:
    source = resolve_lightcurve_source(config)
    job_state = PipelineJobState()

    def worker() -> None:
        log_stream = JobLogBuffer(job_state)
        try:
            result = run_modeling_job(config, source=source, log_stream=log_stream)
        except Exception as exc:
            job_state.mark_failed(exc)
        else:
            job_state.mark_succeeded(result)

    thread = threading.Thread(
        target=worker, name="pymit-streamlit-pipeline", daemon=True
    )
    thread.start()
    return job_state


@st.fragment(run_every=1.0)
def render_running_job() -> None:
    job_state = st.session_state.get("pipeline_job")
    if job_state is None:
        return

    snapshot = job_state.snapshot()
    st.subheader("Run log")
    _render_log_text_area(
        "Pipeline output",
        snapshot["log"] or "Waiting for pipeline output...",
        "running_pipeline_output",
        disabled=True,
    )

    if snapshot["status"] == "running":
        st.info("Pipeline is running...")
        return

    if snapshot["status"] == "failed":
        st.error(snapshot["error"])
        st.session_state["last_pipeline_log"] = snapshot["log"]
        st.session_state.pop("pipeline_job", None)
        return

    if snapshot["status"] == "succeeded":
        st.session_state["last_result"] = snapshot["result"]
        st.session_state.pop("pipeline_job", None)
        st.rerun()


def render_results(result: dict, asteroid_name: str, config: dict) -> None:
    vertices = result["vertices"]
    faces = result["faces"]
    output_dir = result["output_dir"]

    if vertices is not None and faces is not None:
        with st.expander("3D shape", expanded=True):
            st.plotly_chart(build_model_figure(vertices, faces), width="stretch")

    lightcurve_figure = result.get("lightcurve_figure")
    if lightcurve_figure is not None:
        with st.expander("Observed vs modeled phase curve", expanded=False):
            st.plotly_chart(lightcurve_figure, width="stretch")

    base_name = asteroid_name.replace(" ", "_")
    folded_residual_plot = result.get("folded_residual_plot")
    if folded_residual_plot is None:
        folded_residual_plot = output_dir / f"{base_name}_lightcurves_folded_residuals.png"
    else:
        folded_residual_plot = Path(folded_residual_plot)
    if folded_residual_plot.exists():
        with st.expander("Folded residuals by rotation phase", expanded=False):
            st.image(str(folded_residual_plot))

    sky_projection_plot = output_dir / f"{base_name}_sky_projection.png"
    if sky_projection_plot.exists():
        with st.expander("Sky projection", expanded=False):
            st.image(str(sky_projection_plot))

    synthetic_lightcurve_plot = output_dir / f"{base_name}_synthetic_lightcurve.png"
    if synthetic_lightcurve_plot.exists():
        with st.expander("Synthetic lightcurve", expanded=False):
            st.image(str(synthetic_lightcurve_plot))

    period_scan_results_file = output_dir / f"{base_name}_period_scan.csv"
    period_scan_plot = output_dir / f"{base_name}_period_scan.png"
    if period_scan_results_file.exists() or period_scan_plot.exists():
        with st.expander("Period scan", expanded=False):
            if period_scan_plot.exists():
                st.image(str(period_scan_plot))
            if period_scan_results_file.exists():
                period_scan_results = pd.read_csv(period_scan_results_file, nrows=10)
                st.caption("Showing the first 10 period-scan rows. Download the CSV for all results.")
                st.dataframe(period_scan_results)

    scan_results_file = output_dir / f"{base_name}_pole_scan_results.csv"
    scan_best_file = output_dir / f"{base_name}_pole_scan_best.json"
    if scan_results_file.exists():
        with st.expander("Pole grid scan", expanded=False):
            pole_scan_maps = build_pole_scan_map_figure(
                scan_results_file,
                scan_best_file,
                coordinate_mode=config.get("pole_scan_map_coordinate_mode", "initial"),
            )
            for pole_scan_map in pole_scan_maps:
                st.pyplot(pole_scan_map, clear_figure=True)
            scan_results = pd.read_csv(scan_results_file, nrows=10)
            st.caption("Showing the first 10 pole-scan rows. Download the CSV for all results.")
            st.dataframe(scan_results)
            if scan_best_file.exists():
                st.json(json.loads(scan_best_file.read_text()))

    with st.expander("Generated files", expanded=False):
        outputs = collect_generated_outputs(output_dir, asteroid_name)
        if not outputs:
            st.warning("No generated files found.")
            return

        for output in outputs:
            st.download_button(
                label=f"Download {output.label}",
                data=output.path.read_bytes(),
                file_name=output.path.name,
                mime=output.mime_type,
            )


def main() -> None:
    st.set_page_config(page_title="PyMit Asteroid Modeler", layout="wide")
    st.title("PyMit Asteroid Modeler")
    st.caption("Run DAMIT convex inversion and Minkowski shape reconstruction from a local browser UI.")

    _init_session_defaults()
    config = render_sidebar()
    if config["submitted"]:
        existing_job = st.session_state.get("pipeline_job")
        existing_snapshot = existing_job.snapshot() if existing_job is not None else None
        if existing_snapshot is not None and existing_snapshot["status"] == "running":
            st.warning("A modeling pipeline is already running.")
        else:
            _save_current_params()
            try:
                st.session_state["pipeline_job"] = start_modeling_job(config)
                st.session_state.pop("last_pipeline_log", None)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    if st.session_state.get("pipeline_job") is not None:
        render_running_job()
        return

    failed_log = st.session_state.get("last_pipeline_log")
    if failed_log:
        with st.expander("Run log", expanded=False):
            _render_log_text_area(
                "Pipeline output",
                failed_log,
                "failed_pipeline_output",
                disabled=True,
            )

    result = st.session_state.get("last_result")
    if result is None:
        st.info("Configure inputs in the sidebar, then click Run modeling.")
        return

    st.success("Modeling complete.")
    with st.expander("Fit result", expanded=True):
        st.json(result["fit_result"])
    with st.expander("Run log", expanded=False):
        _render_log_text_area(
            "Pipeline output",
            result["log"],
            "final_pipeline_output",
            disabled=True,
        )
    render_results(result, config["asteroid_name"], config)


if __name__ == "__main__":
    main()
