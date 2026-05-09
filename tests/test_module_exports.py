import unittest

import pymit.asteroid_modeling as am


class ModuleExportTests(unittest.TestCase):
    def test_asteroid_modeling_reexports_existing_public_helpers(self):
        expected_names = [
            "AsteroidModelError",
            "AsteroidModeler",
            "run_pipeline",
            "run_convexinv",
            "run_minkowski",
            "plot_model",
            "plot_model_plotly",
            "plot_sky_projection",
            "plot_synthetic_lightcurve",
            "plot_lightcurves",
            "project_shape_to_sky",
            "compute_synthetic_lightcurve",
            "first_observation_jd",
            "dataframe_to_lcs_format",
            "csv_to_lcs_format",
            "save_sky_projection_csv",
            "save_synthetic_lightcurve_csv",
            "create_convexinv_param_file",
            "create_conjgradinv_param_file",
            "load_model_obj",
            "save_model_obj",
            "_parse_minkowski_output",
            "_triangulate_faces",
            "_is_http_url",
            "_download_lightcurve_url",
        ]

        for name in expected_names:
            with self.subTest(name=name):
                self.assertTrue(hasattr(am, name), name)

    def test_top_level_package_exports_modeler_and_error(self):
        import pymit

        self.assertIs(pymit.AsteroidModeler, am.AsteroidModeler)
        self.assertIs(pymit.AsteroidModelError, am.AsteroidModelError)


if __name__ == "__main__":
    unittest.main()
