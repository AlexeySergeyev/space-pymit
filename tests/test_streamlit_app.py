import unittest
from pathlib import Path


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
        self.assertIn("phase_func_a_fixed=False", source)
        self.assertIn("phase_func_d_fixed=False", source)
        self.assertIn("phase_func_k_fixed=False", source)
        self.assertIn("phase_func_c_fixed=False", source)
        self.assertNotIn("Lambert coefficient free", source)
        self.assertNotIn('st.sidebar.header("Input")', source)
        self.assertNotIn('st.sidebar.header("Convex inversion")', source)
        self.assertNotIn('st.sidebar.header("Minkowski reconstruction")', source)

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

    def test_auto_save_and_restore_params(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (repo_root / "apps/streamlit_app.py").read_text()

        self.assertIn("load_saved_params", source)
        self.assertIn("save_params", source)
        self.assertIn("_init_session_defaults", source)
        self.assertIn("_save_current_params", source)
        self.assertIn("_PARAM_DEFAULTS", source)

    def test_results_label_phase_curve(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (repo_root / "apps/streamlit_app.py").read_text()

        self.assertIn('st.subheader("Observed vs modeled phase curve")', source)
        self.assertNotIn('st.subheader("Observed vs modeled lightcurves")', source)


if __name__ == "__main__":
    unittest.main()
