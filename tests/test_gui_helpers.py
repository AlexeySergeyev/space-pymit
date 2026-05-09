import unittest
from datetime import datetime, timezone
from pathlib import Path
import tempfile

import numpy as np

import pymit.gui_helpers as gui_helpers
from pymit.gui_helpers import InversionSettings, build_inversion_options, build_conjgradinv_options
from pymit.gui_helpers import GeneratedOutput, collect_generated_outputs, save_uploaded_lightcurve
from pymit.gui_helpers import build_model_figure, current_julian_date


class GuiHelperOptionTests(unittest.TestCase):
    def test_build_inversion_options_returns_expected_keys(self):
        settings = InversionSettings(
            initial_lambda=220.0,
            initial_lambda_fixed=True,
            initial_beta=0.0,
            initial_beta_fixed=True,
            initial_period=5.76198,
            initial_period_fixed=True,
            phase_func_a=0.5,
            phase_func_a_fixed=False,
            phase_func_d=0.1,
            phase_func_d_fixed=False,
            phase_func_k=-1.05,
            phase_func_k_fixed=False,
            phase_func_c=0.1,
            phase_func_c_fixed=False,
            convexity_regularization=0.1,
            spherical_harmonics_degree=6,
            spherical_harmonics_order=6,
            number_of_rows=8,
            iteration_stop_condition=50,
        )

        options = build_inversion_options(settings)

        self.assertEqual(options["initial_lambda"], 220.0)
        self.assertEqual(options["initial_lambda_fixed"], 1)
        self.assertEqual(options["initial_beta"], 0.0)
        self.assertEqual(options["initial_beta_fixed"], 1)
        self.assertEqual(options["initial_period"], 5.76198)
        self.assertEqual(options["initial_period_fixed"], 1)
        self.assertEqual(options["phase_func_a"], 0.5)
        self.assertEqual(options["phase_func_a_fixed"], 0)
        self.assertEqual(options["phase_func_d"], 0.1)
        self.assertEqual(options["phase_func_d_fixed"], 0)
        self.assertEqual(options["phase_func_k"], -1.05)
        self.assertEqual(options["phase_func_k_fixed"], 0)
        self.assertEqual(options["phase_func_c"], 0.1)
        self.assertEqual(options["phase_func_c_fixed"], 0)
        self.assertEqual(options["convexity_regularization"], 0.1)
        self.assertEqual(options["spherical_harmonics_degree"], 6)
        self.assertEqual(options["spherical_harmonics_order"], 6)
        self.assertEqual(options["number_of_rows"], 8)
        self.assertEqual(options["iteration_stop_condition"], 50)

    def test_build_conjgradinv_options_returns_expected_keys(self):
        options = build_conjgradinv_options(
            convexity_weight=0.2,
            number_of_rows=8,
            number_of_iterations=100,
        )

        self.assertEqual(
            options,
            {
                "convexity_weight": 0.2,
                "number_of_rows": 8,
                "number_of_iterations": 100,
            },
        )


class GuiHelperDateTests(unittest.TestCase):
    def test_current_julian_date_converts_utc_datetime(self):
        self.assertEqual(
            current_julian_date(datetime(1970, 1, 1, tzinfo=timezone.utc)),
            2440587.5,
        )
        self.assertEqual(
            current_julian_date(datetime(2000, 1, 1, 12, tzinfo=timezone.utc)),
            2451545.0,
        )


class FakeUpload:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getbuffer(self):
        return self._data


class GuiHelperFileTests(unittest.TestCase):
    def test_save_and_load_params_round_trips_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_params_file = gui_helpers.PARAMS_FILE
            gui_helpers.PARAMS_FILE = Path(tmp) / "params.json"
            try:
                gui_helpers.save_params({"initial_period": 33.0, "asteroid_name": "A7753"})

                params = gui_helpers.load_saved_params()
            finally:
                gui_helpers.PARAMS_FILE = original_params_file

        self.assertEqual(params, {"initial_period": 33.0, "asteroid_name": "A7753"})

    def test_save_uploaded_lightcurve_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            upload = FakeUpload("input.csv", b"curve_id,jd\n1,2450000.0\n")

            saved = save_uploaded_lightcurve(upload, Path(tmp))

            self.assertEqual(saved.name, "input.csv")
            self.assertEqual(saved.read_bytes(), b"curve_id,jd\n1,2450000.0\n")

    def test_collect_generated_outputs_returns_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "A7753.obj").write_text("obj")
            (output_dir / "A7753_model.html").write_text("<html></html>")
            (output_dir / "A7753_model.png").write_bytes(b"png")
            (output_dir / "A7753_lightcurves.png").write_bytes(b"lc")
            (output_dir / "A7753_areas.txt").write_text("areas")

            outputs = collect_generated_outputs(output_dir, "A7753")

            self.assertEqual(
                outputs,
                [
                    GeneratedOutput("OBJ model", output_dir / "A7753.obj", "text/plain"),
                    GeneratedOutput("Interactive model HTML", output_dir / "A7753_model.html", "text/html"),
                    GeneratedOutput("Static model PNG", output_dir / "A7753_model.png", "image/png"),
                    GeneratedOutput("Lightcurve plot PNG", output_dir / "A7753_lightcurves.png", "image/png"),
                    GeneratedOutput("Areas and normals TXT", output_dir / "A7753_areas.txt", "text/plain"),
                ],
            )


class GuiHelperPlotTests(unittest.TestCase):
    def test_build_model_figure_returns_mesh3d(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        faces = [[1, 2, 3], [1, 2, 4]]

        fig = build_model_figure(vertices, faces)

        self.assertEqual(len(fig.data), 1)
        self.assertEqual(fig.data[0].type, "mesh3d")
        self.assertEqual(list(fig.data[0].i), [0, 0])
        self.assertEqual(list(fig.data[0].j), [1, 1])
        self.assertEqual(list(fig.data[0].k), [2, 3])


if __name__ == "__main__":
    unittest.main()
