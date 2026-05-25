import unittest
from pathlib import Path
import tempfile


class StreamlitAppLayoutTests(unittest.TestCase):
    def test_sidebar_uses_collapsible_sections(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (repo_root / "apps/streamlit_app.py").read_text()

        self.assertIn('st.sidebar.expander("Input", expanded=True)', source)
        self.assertIn('st.sidebar.expander("Convex inversion", expanded=False)', source)
        self.assertIn(
            'st.sidebar.expander("Minkowski reconstruction", expanded=False)', source
        )
        self.assertIn(
            'st.sidebar.expander("Sky projection and synthetic lightcurve", expanded=False)',
            source,
        )
        self.assertIn('st.sidebar.expander("Pole grid scan", expanded=False)', source)
        self.assertIn('st.sidebar.expander("Period scan", expanded=False)', source)
        self.assertIn('"Period scan"', source)
        self.assertIn('"Period start [hours]"', source)
        self.assertIn('"Period end [hours]"', source)
        self.assertIn('"Period interval coefficient"', source)
        self.assertIn('"Period scan minimum iterations"', source)
        self.assertIn('"Period scan workers"', source)
        self.assertIn('"period_scan_spherical_harmonics_degree": 3', source)
        self.assertIn('"period_scan_spherical_harmonics_order": 3', source)
        self.assertIn('"period_scan_number_of_rows": 4', source)
        self.assertIn('"Pole grid search"', source)
        self.assertIn('"Golden spiral N"', source)
        self.assertIn('"Pole scan workers"', source)
        self.assertIn('"Plot map by fitted pole values"', source)
        self.assertIn('st.number_input("Initial lambda [deg]", key="initial_lambda")', source)
        self.assertIn('st.number_input("Initial beta [deg]", key="initial_beta")', source)
        self.assertIn('"LSL p1 Lambert coefficient c"', source)
        self.assertIn('"LSL p2 phase amplitude a"', source)
        self.assertIn('"LSL p3 phase width d"', source)
        self.assertIn('"LSL p4 phase slope k"', source)
        self.assertIn("DAMIT p1", source)
        self.assertIn("DAMIT p2", source)
        self.assertIn("DAMIT p3", source)
        self.assertIn("DAMIT p4", source)
        self.assertIn('"phase_func_a_fixed": False', source)
        self.assertIn('"phase_func_d_fixed": False', source)
        self.assertIn('"phase_func_k_fixed": False', source)
        self.assertIn('"Fit p2"', source)
        self.assertIn('"Fit p3"', source)
        self.assertIn('"Fit p4"', source)
        self.assertIn('key="phase_func_a_fixed"', source)
        self.assertIn('key="phase_func_d_fixed"', source)
        self.assertIn('key="phase_func_k_fixed"', source)
        self.assertIn("can cause singular matrix errors", source)
        self.assertIn("phase_func_c_fixed=False", source)
        self.assertNotIn("Lambert coefficient free", source)
        self.assertNotIn('st.sidebar.header("Input")', source)
        self.assertNotIn('st.sidebar.header("Convex inversion")', source)
        self.assertNotIn('st.sidebar.header("Minkowski reconstruction")', source)

    def test_sidebar_stage_checkboxes_are_outside_collapsible_sections(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (repo_root / "apps/streamlit_app.py").read_text()

        convex_checkbox = source.index(
            "run_convex_inversion = st.sidebar.checkbox("
        )
        convex_expander = source.index(
            'st.sidebar.expander("Convex inversion", expanded=False)'
        )
        minkowski_checkbox = source.index(
            "run_minkowski_reconstruction = st.sidebar.checkbox("
        )
        minkowski_expander = source.index(
            'st.sidebar.expander("Minkowski reconstruction", expanded=False)'
        )
        pole_checkbox = source.index(
            'st.sidebar.checkbox("Pole grid search", key="run_pole_grid_scan")'
        )
        pole_expander = source.index(
            'st.sidebar.expander("Pole grid scan", expanded=False)'
        )
        period_checkbox = source.index(
            'st.sidebar.checkbox("Period scan", key="run_period_scan")'
        )
        period_expander = source.index(
            'st.sidebar.expander("Period scan", expanded=False)'
        )
        sky_checkbox = source.index(
            "generate_projection_products = st.sidebar.checkbox("
        )
        sky_expander = source.index(
            'st.sidebar.expander("Sky projection and synthetic lightcurve", expanded=False)'
        )

        self.assertLess(convex_checkbox, convex_expander)
        self.assertLess(period_checkbox, period_expander)
        self.assertLess(period_checkbox, convex_checkbox)
        self.assertLess(period_expander, convex_checkbox)
        self.assertLess(minkowski_checkbox, minkowski_expander)
        self.assertLess(pole_checkbox, pole_expander)
        self.assertLess(sky_checkbox, sky_expander)
        self.assertNotIn('"Run pole grid scan"', source)
        self.assertNotIn('"Generate projection outputs"', source)
        self.assertNotIn('st.session_state["run_convex_inversion"] = True', source)
        self.assertNotIn('st.session_state["run_minkowski_reconstruction"] = True', source)
        self.assertNotIn('disabled=True,\n        help="Required for the modeling pipeline."', source)
        self.assertNotIn('disabled=True,\n        help="Required to reconstruct the 3D shape model."', source)

    def test_plotly_chart_uses_streamlit_width_api(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (repo_root / "apps/streamlit_app.py").read_text()

        self.assertIn("st.plotly_chart(build_model_figure(vertices, faces), width=\"stretch\")", source)
        self.assertNotIn("use_container_width", source)

    def test_modeling_job_generates_projection_products(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (repo_root / "apps/streamlit_app.py").read_text()

        self.assertIn("modeler.plot_sky_projection(", source)
        self.assertIn("modeler.plot_synthetic_lightcurve(", source)
        self.assertIn('"generate_projection_products"', source)
        self.assertIn("current_julian_date()", source)
        self.assertIn('"projection_jd"', source)
        self.assertNotIn('"projection_period_hours"', source)
        self.assertNotIn('"projection_zero_time"', source)
        self.assertNotIn('"projection_initial_rotation_angle"', source)

    def test_modeling_job_logs_post_convexinv_stages(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (repo_root / "apps/streamlit_app.py").read_text()

        self.assertIn('print("Running pole grid scan...', source)
        self.assertIn('print("Plotting observed vs modeled lightcurves...", flush=True)', source)
        self.assertIn('print("Plotting folded residual lightcurve...", flush=True)', source)
        self.assertIn('print("Plotting static 3D model...", flush=True)', source)
        self.assertIn('print("Generating interactive 3D model...", flush=True)', source)
        self.assertIn('print("Exporting OBJ model...", flush=True)', source)
        self.assertIn('print("Generating sky projection...", flush=True)', source)
        self.assertIn('print("Generating synthetic lightcurve...", flush=True)', source)

    def test_auto_save_and_restore_params(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (repo_root / "apps/streamlit_app.py").read_text()

        self.assertIn("load_saved_params", source)
        self.assertIn("save_params", source)
        self.assertIn("_init_session_defaults", source)
        self.assertIn("_save_current_params", source)
        self.assertIn("_PARAM_DEFAULTS", source)

    def test_params_are_saved_when_run_starts_not_when_worker_finishes(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (repo_root / "apps/streamlit_app.py").read_text()

        submit_index = source.index('if config["submitted"]:')
        start_index = source.index('st.session_state["pipeline_job"] = start_modeling_job(config)')
        save_index = source.index("_save_current_params()", submit_index)
        rerun_index = source.index("st.rerun()", start_index)
        success_index = source.index('if snapshot["status"] == "succeeded":')
        success_block = source[success_index:source.index("def render_results", success_index)]

        self.assertLess(save_index, start_index)
        self.assertLess(start_index, rerun_index)
        self.assertNotIn("_save_current_params()", success_block)

    def test_results_label_phase_curve(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (repo_root / "apps/streamlit_app.py").read_text()

        self.assertIn(
            'with st.expander("Observed vs modeled phase curve", expanded=False):',
            source,
        )
        self.assertIn('st.plotly_chart(lightcurve_figure, width="stretch")', source)
        self.assertIn(
            'with st.expander("Folded residuals by rotation phase", expanded=False):',
            source,
        )
        self.assertIn("st.image(str(folded_residual_plot))", source)
        self.assertNotIn('st.subheader("Observed vs modeled lightcurves")', source)

    def test_results_are_grouped_in_collapsible_sections(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (repo_root / "apps/streamlit_app.py").read_text()

        self.assertIn('with st.expander("Fit result", expanded=True):', source)
        self.assertIn('with st.expander("Run log", expanded=False):', source)
        self.assertIn('with st.expander("3D shape", expanded=True):', source)
        self.assertIn(
            'with st.expander("Observed vs modeled phase curve", expanded=False):',
            source,
        )
        self.assertIn(
            'with st.expander("Folded residuals by rotation phase", expanded=False):',
            source,
        )
        self.assertIn('with st.expander("Sky projection", expanded=False):', source)
        self.assertIn('with st.expander("Synthetic lightcurve", expanded=False):', source)
        self.assertIn('with st.expander("Pole grid scan", expanded=False):', source)
        self.assertIn('with st.expander("Generated files", expanded=False):', source)

    def test_modeling_job_dispatches_to_pole_scan_when_enabled(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (repo_root / "apps/streamlit_app.py").read_text()

        self.assertIn("run_pole_grid_scan = config.get", source)
        self.assertIn("if run_pole_grid_scan:", source)
        self.assertIn("run_period_scan = config.get", source)
        self.assertIn("if run_period_scan:", source)
        self.assertIn("modeler.run_period_scan(", source)
        self.assertIn('period_scan_options=config["period_scan_options"]', source)
        self.assertIn('workers=config["period_scan_workers"]', source)
        self.assertIn("modeler.run_pole_grid_scan(", source)
        self.assertIn('n=config["pole_grid_n"]', source)
        self.assertIn('workers=config["pole_grid_workers"]', source)
        self.assertIn('with st.expander("Pole grid scan", expanded=False):', source)
        self.assertIn('with st.expander("Period scan", expanded=False):', source)
        self.assertIn('f"{base_name}_period_scan.csv"', source)
        self.assertIn('f"{base_name}_period_scan.png"', source)
        self.assertIn('coordinate_mode=config.get("pole_scan_map_coordinate_mode", "initial")', source)
        self.assertIn("for pole_scan_map in pole_scan_maps:", source)
        self.assertIn("st.pyplot(pole_scan_map, clear_figure=True)", source)
        self.assertIn("pd.read_csv(scan_results_file, nrows=10)", source)
        self.assertIn('st.dataframe(scan_results)', source)

    def test_modeling_job_can_run_period_scan_only(self):
        import importlib.util

        repo_root = Path(__file__).resolve().parents[1]
        module_path = repo_root / "apps" / "streamlit_app.py"
        spec = importlib.util.spec_from_file_location(
            "streamlit_app_for_period_scan_only_tests", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        stage_events = []

        class FakeModeler:
            def __init__(self, asteroid_name, output_dir):
                self.asteroid_name = asteroid_name
                self.output_dir = Path(output_dir)
                self.fit_result = None
                self.lightcurve_file = str(self.output_dir / "input.lc.txt")
                self.lightcurves = None

            def load_lightcurves(self, source):
                stage_events.append(("load", source))

            def load_parameters(self, inversion_json, conjgradinv_json):
                stage_events.append(("params", inversion_json, conjgradinv_json))

            def run_period_scan(self, period_scan_options, workers, verbose=False):
                stage_events.append(("period_scan", period_scan_options, workers, verbose))
                self.fit_result = {"initial_period_from_period_scan": 6.0}
                return object()

            def run_convex_inversion(self, verbose=False):
                stage_events.append(("convex", verbose))
                raise AssertionError("convexinv should not run")

            def run_minkowski_reconstruction(self, verbose=False):
                stage_events.append(("minkowski", verbose))
                raise AssertionError("minkowski should not run")

            def plot_lightcurves_results(self, save, show, max_curves):
                raise AssertionError("lightcurve plotting should not run")

            def plot_model(self, save, show):
                raise AssertionError("model plotting should not run")

            def plot_model_plotly(self, save, show):
                raise AssertionError("model plotting should not run")

            def export_obj(self):
                raise AssertionError("OBJ export should not run")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            config = {
                "asteroid_name": "A7753",
                "output_dir": output_dir,
                "inversion_options": {"initial_period": 12.0},
                "conjgradinv_options": {"number_of_iterations": 3},
                "run_period_scan": True,
                "period_scan_options": {"period_start": 5.0, "period_end": 7.0},
                "period_scan_workers": 2,
                "run_convex_inversion": False,
                "run_minkowski_reconstruction": False,
                "run_pole_grid_scan": False,
                "generate_projection_products": False,
            }
            module.AsteroidModeler = FakeModeler

            result = module.run_modeling_job(
                config,
                source="source.lc.txt",
                log_stream=module.JobLogBuffer(module.PipelineJobState()),
            )

        self.assertIn(
            ("period_scan", {"period_start": 5.0, "period_end": 7.0}, 2, True),
            stage_events,
        )
        self.assertNotIn(("convex", True), stage_events)
        self.assertNotIn(("minkowski", True), stage_events)
        self.assertIsNone(result["vertices"])
        self.assertIsNone(result["faces"])

    def test_modeling_job_uses_background_thread_and_polling_fragment(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (repo_root / "apps/streamlit_app.py").read_text()

        self.assertIn("class PipelineJobState", source)
        self.assertIn("class JobLogBuffer", source)
        self.assertIn("def start_modeling_job", source)
        self.assertIn("threading.Thread", source)
        self.assertIn('name="pymit-streamlit-pipeline"', source)
        self.assertIn("@st.fragment(run_every=1.0)", source)
        self.assertIn("st.rerun()", source)
        self.assertNotIn("class StreamlitLogBuffer", source)
        self.assertNotIn("log_placeholder", source)
        self.assertNotIn("st.code(", source)
        self.assertIn("_render_log_text_area(", source)
        self.assertIn("_log_text_area_key(", source)
        self.assertNotIn('key="running_pipeline_output"', source)
        self.assertNotIn('key="failed_pipeline_output"', source)
        self.assertNotIn('key="final_pipeline_output"', source)
        self.assertNotIn("log_stream = io.StringIO()", source)


class StreamlitJobStateTests(unittest.TestCase):
    def test_generate_folded_residual_plot_writes_png_from_model_period(self):
        import importlib.util
        import types

        repo_root = Path(__file__).resolve().parents[1]
        module_path = repo_root / "apps" / "streamlit_app.py"
        spec = importlib.util.spec_from_file_location(
            "streamlit_app_for_folded_residual_test", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            lightcurve_file = output_dir / "input.lc.txt"
            model_file = output_dir / "A7753_lc_output.csv"
            param_file = output_dir / "A7753_input_convexinv.txt"
            lightcurve_file.write_text(
                "1\n"
                "3 0\n"
                "2450000.0 10.0 1 0 0 0 1 0\n"
                "2450000.5 12.0 1 0 0 0 1 0\n"
                "2450001.0 11.0 1 0 0 0 1 0\n"
            )
            model_file.write_text("9.5\n12.5\n10.0\n")
            param_file.write_text(
                "1\t0\tinital lambda [deg] (0/1 - fixed/free)\n"
                "2\t0\tinitial beta [deg] (0/1 - fixed/free)\n"
                "12\t0\tinital period [hours] (0/1 - fixed/free)\n"
                "2450000.25\t\tzero time [JD]\n"
            )
            modeler = types.SimpleNamespace(
                fit_result={"period": 12.0},
                lightcurve_file=str(lightcurve_file),
                lightcurves=None,
            )

            plot_path = module.generate_folded_residual_plot(
                modeler, output_dir, "A7753", max_curves=1
            )

            self.assertEqual(
                plot_path, output_dir / "A7753_lightcurves_folded_residuals.png"
            )
            self.assertTrue(plot_path.exists())

    def test_job_log_buffer_appends_to_thread_safe_state(self):
        import importlib.util

        repo_root = Path(__file__).resolve().parents[1]
        module_path = repo_root / "apps" / "streamlit_app.py"
        spec = importlib.util.spec_from_file_location("streamlit_app_for_tests", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        job = module.PipelineJobState()
        buffer = module.JobLogBuffer(job)

        buffer.write("first line\n")
        buffer.write("second line\n")
        buffer.flush()

        snapshot = job.snapshot()
        self.assertEqual(snapshot["status"], "running")
        self.assertEqual(snapshot["log"], "first line\nsecond line\n")
        self.assertIsNone(snapshot["result"])
        self.assertIsNone(snapshot["error"])

    def test_log_text_area_key_changes_when_log_content_changes(self):
        import importlib.util

        repo_root = Path(__file__).resolve().parents[1]
        module_path = repo_root / "apps" / "streamlit_app.py"
        spec = importlib.util.spec_from_file_location("streamlit_app_for_log_key_tests", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        waiting_key = module._log_text_area_key(
            "running_pipeline_output", "Waiting for pipeline output..."
        )
        convex_key = module._log_text_area_key(
            "running_pipeline_output", "Running convex inversion...\n"
        )
        minkowski_key = module._log_text_area_key(
            "running_pipeline_output",
            "Running convex inversion...\nConvex inversion complete.\nRunning Minkowski reconstruction...\n",
        )

        self.assertNotEqual(waiting_key, convex_key)
        self.assertNotEqual(convex_key, minkowski_key)
        self.assertTrue(convex_key.startswith("running_pipeline_output_"))

    def test_streamlit_log_text_area_rerenders_when_log_content_changes(self):
        from streamlit.testing.v1 import AppTest

        repo_root = Path(__file__).resolve().parents[1]
        script = f"""
import importlib.util
from pathlib import Path
import sys

import streamlit as st

repo_root = Path({str(repo_root)!r})
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

spec = importlib.util.spec_from_file_location(
    "streamlit_app_log_widget_test", repo_root / "apps" / "streamlit_app.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

st.session_state.setdefault("log_value", "Waiting for pipeline output...")
module._render_log_text_area(
    "Pipeline output",
    st.session_state["log_value"],
    "running_pipeline_output",
    disabled=True,
        )
"""
        app = AppTest.from_string(script)
        app.run(timeout=15)
        self.assertEqual(app.text_area[0].value, "Waiting for pipeline output...")

        app.session_state["log_value"] = "Running convex inversion...\n"
        app.run(timeout=15)
        self.assertEqual(app.text_area[0].value, "Running convex inversion...\n")

        app.session_state["log_value"] = (
            "Running convex inversion...\n"
            "Convex inversion complete.\n"
            "Running Minkowski reconstruction...\n"
        )
        app.run(timeout=15)
        self.assertEqual(
            app.text_area[0].value,
            "Running convex inversion...\n"
            "Convex inversion complete.\n"
            "Running Minkowski reconstruction...\n",
        )

    def test_modeling_job_logs_convex_completion_before_minkowski_starts(self):
        import importlib.util
        import types

        repo_root = Path(__file__).resolve().parents[1]
        module_path = repo_root / "apps" / "streamlit_app.py"
        spec = importlib.util.spec_from_file_location("streamlit_app_for_stage_tests", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        stage_events = []

        class FakeModeler:
            def __init__(self, asteroid_name, output_dir):
                self.asteroid_name = asteroid_name
                self.output_dir = Path(output_dir)
                self.fit_result = {"period": 12.0}
                self.lightcurve_file = str(self.output_dir / "input.lc.txt")
                self.lightcurves = None

            def load_lightcurves(self, source):
                stage_events.append(("load", source))

            def load_parameters(self, inversion_json, conjgradinv_json):
                stage_events.append(("params", inversion_json, conjgradinv_json))

            def run_convex_inversion(self, verbose=False):
                stage_events.append(("convex", verbose))
                print("fake convex details", flush=True)
                return self.fit_result

            def run_minkowski_reconstruction(self, verbose=False):
                stage_events.append(("minkowski", verbose))
                print("fake minkowski details", flush=True)
                return [[0.0, 0.0, 0.0]], [[1]]

            def plot_lightcurves_results(self, save, show, max_curves):
                return "lightcurve figure"

            def plot_model(self, save, show):
                pass

            def plot_model_plotly(self, save, show):
                pass

            def export_obj(self):
                pass

        def fake_generate_folded_residual_plot(*args, **kwargs):
            return None

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "input.lc.txt").write_text("1\n")
            config = {
                "asteroid_name": "A7753",
                "output_dir": output_dir,
                "inversion_options": {"initial_period": 12.0},
                "conjgradinv_options": {"number_of_iterations": 3},
                "run_pole_grid_scan": False,
                "generate_projection_products": False,
            }
            module.AsteroidModeler = FakeModeler
            module.generate_folded_residual_plot = fake_generate_folded_residual_plot

            result = module.run_modeling_job(
                config,
                source="source.lc.txt",
                log_stream=module.JobLogBuffer(module.PipelineJobState()),
            )

        log = result["log"]
        self.assertLess(
            log.index("Convex inversion complete."),
            log.index("Running Minkowski reconstruction..."),
        )
        self.assertLess(
            stage_events.index(("convex", True)),
            stage_events.index(("minkowski", True)),
        )

    def test_job_state_records_success_and_failure_snapshots(self):
        import importlib.util

        repo_root = Path(__file__).resolve().parents[1]
        module_path = repo_root / "apps" / "streamlit_app.py"
        spec = importlib.util.spec_from_file_location("streamlit_app_for_tests_state", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        success_job = module.PipelineJobState()
        result = {"fit_result": {"period": 33.0}}
        success_job.mark_succeeded(result)
        success_snapshot = success_job.snapshot()

        self.assertEqual(success_snapshot["status"], "succeeded")
        self.assertEqual(success_snapshot["result"], result)
        self.assertIsNone(success_snapshot["error"])
        self.assertIsNotNone(success_snapshot["finished_at"])

        failed_job = module.PipelineJobState()
        failed_job.mark_failed(RuntimeError("convexinv failed"))
        failed_snapshot = failed_job.snapshot()

        self.assertEqual(failed_snapshot["status"], "failed")
        self.assertIn("convexinv failed", failed_snapshot["error"])
        self.assertIn("Pipeline execution failed: convexinv failed", failed_snapshot["log"])
        self.assertIsNone(failed_snapshot["result"])
        self.assertIsNotNone(failed_snapshot["finished_at"])

        prefixed_job = module.PipelineJobState()
        prefixed_job.mark_failed(
            RuntimeError("Pipeline execution failed: minkowski crashed")
        )
        prefixed_snapshot = prefixed_job.snapshot()

        self.assertEqual(
            prefixed_snapshot["error"], "Pipeline execution failed: minkowski crashed"
        )
        self.assertEqual(
            prefixed_snapshot["log"].count("Pipeline execution failed:"), 1
        )


if __name__ == "__main__":
    unittest.main()
