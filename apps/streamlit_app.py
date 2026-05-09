from pathlib import Path
import contextlib
import io
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
    "phase_func_d": 0.1,
    "phase_func_k": -1.05,
    "phase_func_c": 0.1,
    "convexity_regularization": 0.1,
    "spherical_harmonics_degree": 6,
    "spherical_harmonics_order": 6,
    "number_of_rows": 8,
    "iteration_stop_condition": 50,
    "convexity_weight": 0.2,
    "reconstruction_iterations": 100,
    "generate_projection_products": True,
    "synthetic_lightcurve_steps": 72,
}

_PERSISTENT_KEYS = list(_PARAM_DEFAULTS.keys()) + ["projection_jd"]


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
            phase_func_a_fixed=False,
            phase_func_d=st.number_input(
                "LSL p3 phase width d",
                min_value=0.0,
                format="%.4f",
                help="DAMIT p3 for the LSL scattering model. Phase function width d.",
                key="phase_func_d",
            ),
            phase_func_d_fixed=False,
            phase_func_k=st.number_input(
                "LSL p4 phase slope k",
                format="%.4f",
                help="DAMIT p4 for the LSL scattering model. Phase function linear slope k.",
                key="phase_func_k",
            ),
            phase_func_k_fixed=False,
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

    with st.sidebar.expander("Minkowski reconstruction", expanded=False):
        convexity_weight = st.number_input(
            "Convexity weight", format="%.4f", key="convexity_weight"
        )
        reconstruction_iterations = st.number_input(
            "Reconstruction iterations", min_value=1, step=1, key="reconstruction_iterations"
        )

    with st.sidebar.expander("Sky projection and synthetic lightcurve", expanded=False):
        generate_projection_products = st.checkbox(
            "Generate projection outputs", key="generate_projection_products"
        )
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
        "generate_projection_products": generate_projection_products,
        "projection_jd": projection_jd,
        "synthetic_lightcurve_steps": synthetic_lightcurve_steps,
        "inversion_options": build_inversion_options(settings),
        "conjgradinv_options": build_conjgradinv_options(
            convexity_weight=convexity_weight,
            number_of_rows=settings.number_of_rows,
            number_of_iterations=reconstruction_iterations,
        ),
    }


def run_modeling_job(config: dict) -> dict:
    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    modeler = AsteroidModeler(
        asteroid_name=config["asteroid_name"],
        output_dir=str(output_dir),
    )

    if config["input_mode"] == "DAMIT URL":
        if not config["damit_url"].strip():
            raise ValueError("DAMIT URL is required.")
        source = config["damit_url"].strip()
    else:
        if config["uploaded_file"] is None:
            raise ValueError("Upload a CSV or native DAMIT text file.")
        source = str(save_uploaded_lightcurve(config["uploaded_file"], output_dir))

    log_stream = io.StringIO()
    with contextlib.redirect_stdout(log_stream), contextlib.redirect_stderr(log_stream):
        modeler.load_lightcurves(source)
        modeler.load_parameters(
            inversion_json=config["inversion_options"],
            conjgradinv_json=config["conjgradinv_options"],
        )
        vertices, faces = modeler.run_inversion(verbose=True)
        modeler.plot_lightcurves_results(save=True, show=False, max_curves=3)
        modeler.plot_model(save=True, show=False)
        modeler.plot_model_plotly(save=True, show=False)
        modeler.export_obj()
        if config["generate_projection_products"]:
            modeler.plot_sky_projection(
                save=True,
                show=False,
                jd=config["projection_jd"],
            )
            modeler.plot_synthetic_lightcurve(
                save=True,
                show=False,
                n_steps=config["synthetic_lightcurve_steps"],
                jd=config["projection_jd"],
            )

    return {
        "modeler": modeler,
        "vertices": vertices,
        "faces": faces,
        "fit_result": modeler.fit_result,
        "log": log_stream.getvalue(),
        "output_dir": output_dir,
    }


def render_results(result: dict, asteroid_name: str) -> None:
    vertices = result["vertices"]
    faces = result["faces"]
    output_dir = result["output_dir"]

    st.subheader("3D shape")
    st.plotly_chart(build_model_figure(vertices, faces), width="stretch")

    lightcurve_plot = output_dir / f"{asteroid_name.replace(' ', '_')}_lightcurves.png"
    if lightcurve_plot.exists():
        st.subheader("Observed vs modeled phase curve")
        st.image(str(lightcurve_plot))

    sky_projection_plot = output_dir / f"{asteroid_name.replace(' ', '_')}_sky_projection.png"
    if sky_projection_plot.exists():
        st.subheader("Sky projection")
        st.image(str(sky_projection_plot))

    synthetic_lightcurve_plot = output_dir / f"{asteroid_name.replace(' ', '_')}_synthetic_lightcurve.png"
    if synthetic_lightcurve_plot.exists():
        st.subheader("Synthetic lightcurve")
        st.image(str(synthetic_lightcurve_plot))

    st.subheader("Generated files")
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
        try:
            with st.spinner("Running asteroid modeling pipeline..."):
                st.session_state["last_result"] = run_modeling_job(config)
                _save_current_params()
        except Exception as exc:
            st.error(str(exc))

    result = st.session_state.get("last_result")
    if result is None:
        st.info("Configure inputs in the sidebar, then click Run modeling.")
        return

    st.success("Modeling complete.")
    st.subheader("Fit result")
    st.json(result["fit_result"])
    st.subheader("Run log")
    st.text_area("Pipeline output", value=result["log"], height=280)
    render_results(result, config["asteroid_name"])


if __name__ == "__main__":
    main()
