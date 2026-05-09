import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd

from pymit.projection import (
    compute_synthetic_lightcurve,
    first_observation_jd,
    plot_sky_projection,
    plot_synthetic_lightcurve,
    project_shape_to_sky,
    save_sky_projection_csv,
    save_synthetic_lightcurve_csv,
)
from pymit.asteroid_modeling import AsteroidModeler


VERTICES = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
FACES = [[1, 2, 3], [1, 2, 4]]


class SkyProjectionTests(unittest.TestCase):
    def test_project_shape_to_sky_rotates_vertices_and_marks_visible_faces(self):
        result = project_shape_to_sky(
            VERTICES,
            FACES,
            phase_degrees=90.0,
            view_vector=(0.0, 0.0, 1.0),
        )

        np.testing.assert_allclose(result.projected_vertices[1], [0.0, 1.0], atol=1e-12)
        self.assertEqual(result.visible_faces, [0])
        self.assertEqual(result.face_count, 2)

    def test_save_sky_projection_csv_writes_projected_vertices(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_file = Path(tmp) / "projection.csv"
            result = project_shape_to_sky(VERTICES, FACES)

            save_sky_projection_csv(result, str(output_file))
            text = output_file.read_text()

        self.assertIn("vertex_index,sky_x,sky_y,rotated_x,rotated_y,rotated_z", text)
        self.assertIn("1,0.0,0.0,0.0,0.0,0.0", text)

    def test_project_shape_to_sky_can_use_julian_date_for_phase(self):
        result = project_shape_to_sky(
            VERTICES,
            FACES,
            jd=2450000.25,
            period_hours=24.0,
            zero_time=2450000.0,
            initial_rotation_angle=10.0,
        )

        self.assertAlmostEqual(result.phase_degrees, 100.0)

    def test_plot_sky_projection_writes_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_file = Path(tmp) / "projection.png"

            plot_sky_projection(VERTICES, FACES, save_path=str(output_file), show=False)

            self.assertTrue(output_file.exists())
            self.assertGreater(output_file.stat().st_size, 0)

    def test_plot_sky_projection_draws_hidden_silhouette_and_shaded_visible_facets(self):
        with patch("pymit.projection.PolyCollection", wraps=PolyCollection) as collection_mock:
            plot_sky_projection(VERTICES, FACES, show=False)

        self.assertGreaterEqual(collection_mock.call_count, 2)
        hidden_kwargs = collection_mock.call_args_list[0].kwargs
        visible_kwargs = collection_mock.call_args_list[1].kwargs
        self.assertEqual(hidden_kwargs["facecolors"], "#1f2937")
        self.assertNotEqual(visible_kwargs["facecolors"], "lightgray")
        self.assertTrue(all(mcolors.is_color_like(color) for color in visible_kwargs["facecolors"]))

    def test_first_observation_jd_reads_native_lightcurve_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            lightcurve_file = Path(tmp) / "input.lc.txt"
            lightcurve_file.write_text(
                "1\n"
                "2 0\n"
                "2456877.703423 1.0 0 0 1 0 1 0\n"
                "2456878.027575 1.1 0 0 1 0 1 0\n"
            )

            self.assertEqual(first_observation_jd(str(lightcurve_file)), 2456877.703423)

    def test_first_observation_jd_reads_dataframe_without_rounding(self):
        lightcurves = pd.DataFrame(
            {
                "jd": [2456877.703423, 2456878.027575],
                "brightness": [1.0, 1.1],
            }
        )

        self.assertEqual(first_observation_jd(lightcurves), 2456877.703423)


class SyntheticLightcurveTests(unittest.TestCase):
    def test_compute_synthetic_lightcurve_returns_normalized_brightness(self):
        curve = compute_synthetic_lightcurve(
            VERTICES,
            FACES,
            n_steps=8,
            sun_vector=(0.0, 0.0, 1.0),
            view_vector=(0.0, 0.0, 1.0),
        )

        self.assertEqual(list(curve.columns), ["phase", "phase_degrees", "brightness"])
        self.assertEqual(len(curve), 8)
        self.assertAlmostEqual(curve["brightness"].max(), 1.0)
        self.assertTrue((curve["brightness"] >= 0.0).all())

    def test_compute_synthetic_lightcurve_can_start_from_julian_date_phase(self):
        curve = compute_synthetic_lightcurve(
            VERTICES,
            FACES,
            n_steps=4,
            jd=2450000.25,
            period_hours=24.0,
            zero_time=2450000.0,
            initial_rotation_angle=10.0,
        )

        self.assertAlmostEqual(curve.iloc[0]["phase_degrees"], 100.0)
        self.assertAlmostEqual(curve.iloc[1]["phase_degrees"], 190.0)

    def test_save_and_plot_synthetic_lightcurve_outputs_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_file = Path(tmp) / "synthetic_lightcurve.csv"
            png_file = Path(tmp) / "synthetic_lightcurve.png"
            curve = compute_synthetic_lightcurve(VERTICES, FACES, n_steps=8)

            save_synthetic_lightcurve_csv(curve, str(csv_file))
            plot_synthetic_lightcurve(curve, save_path=str(png_file), show=False)

            self.assertIn("phase,phase_degrees,brightness", csv_file.read_text())
            self.assertTrue(png_file.exists())
            self.assertGreater(png_file.stat().st_size, 0)


class AsteroidModelerProjectionTests(unittest.TestCase):
    def test_modeler_writes_projection_and_synthetic_lightcurve_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            modeler = AsteroidModeler("Demo", output_dir=tmp)
            modeler.vertices = VERTICES
            modeler.faces = FACES
            modeler.inversion_options = {"phase_func_c": 0.1}
            modeler.fit_result = {"period": 24.0}
            modeler.lightcurve_file = str(Path(tmp) / "input.lc.txt")
            Path(modeler.lightcurve_file).write_text(
                "1\n"
                "1 0\n"
                "2450000.0 1.0 0 0 1 0 1 0\n"
            )

            projection = modeler.plot_sky_projection(
                save=True, show=False, jd=2450000.25
            )
            lightcurve = modeler.plot_synthetic_lightcurve(
                save=True, show=False, n_steps=8, jd=2450000.25
            )

            self.assertEqual(projection.face_count, 2)
            self.assertAlmostEqual(projection.phase_degrees, 90.0)
            self.assertAlmostEqual(lightcurve.iloc[0]["phase_degrees"], 90.0)
            self.assertEqual(len(lightcurve), 8)
            self.assertTrue((Path(tmp) / "Demo_sky_projection.png").exists())
            self.assertTrue((Path(tmp) / "Demo_sky_projection.csv").exists())
            self.assertTrue((Path(tmp) / "Demo_synthetic_lightcurve.png").exists())
            self.assertTrue((Path(tmp) / "Demo_synthetic_lightcurve.csv").exists())


if __name__ == "__main__":
    unittest.main()
