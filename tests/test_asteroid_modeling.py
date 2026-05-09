import tempfile
import unittest
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import requests

import pymit.asteroid_modeling as am


NATIVE_LIGHTCURVE = """1
2 0
2456877.703423 6.364793e+00 2.89296193e+00 7.63351006e-01 -5.15297476e-02 3.62072375e+00 4.24643057e-02 -5.15790714e-02
2456878.027575 4.488047e+00 2.89198526e+00 7.66556461e-01 -5.14116706e-02 3.62359570e+00 4.96599899e-02 -5.14486989e-02
"""


class FakeRequestsResponse:
    def __init__(self, payload: str):
        self.content = payload.encode()

    def raise_for_status(self):
        return None


class AsteroidModelingInputTests(unittest.TestCase):
    def test_modeler_accepts_native_damit_lightcurve_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lightcurve_file = tmp_path / "A7753.lc.txt"
            lightcurve_file.write_text(NATIVE_LIGHTCURVE)

            used_lightcurve_files = []

            def fake_run_convexinv(param_file, lightcurve_file, output_areas_file, output_lc_file, verbose=False):
                used_lightcurve_files.append(lightcurve_file)
                Path(output_areas_file).write_text("areas")
                Path(output_lc_file).write_text("1.0\n2.0\n")
                return {"period": 1.23}

            with patch.object(am, "run_convexinv", fake_run_convexinv), patch.object(
                am, "run_minkowski", lambda *args, **kwargs: (np.zeros((1, 3)), [[1]])
            ):
                modeler = am.AsteroidModeler(asteroid_name="A7753", output_dir=str(tmp_path / "out"))
                vertices, faces = modeler.load_lightcurves(str(lightcurve_file)).run_inversion()

            self.assertEqual(used_lightcurve_files, [str(lightcurve_file)])
            self.assertEqual(vertices.shape, (1, 3))
            self.assertEqual(faces, [[1]])
            self.assertEqual(modeler.fit_result, {"period": 1.23})

    def test_run_pipeline_downloads_damit_url_to_native_lightcurve_file(self):
        url = "https://damit.cuni.cz/projects/damit/LightCurves/exportAllForAsteroid/7753/plaintext/A7753.lc.txt"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            downloaded_files = []
            used_lightcurve_files = []

            def fake_get(source_url, timeout=None, verify=True):
                self.assertEqual(source_url, url)
                self.assertEqual(timeout, 30)
                self.assertTrue(verify)
                return FakeRequestsResponse(NATIVE_LIGHTCURVE)

            def fake_run_convexinv(param_file, lightcurve_file, output_areas_file, output_lc_file, verbose=False):
                used_lightcurve_files.append(lightcurve_file)
                downloaded_files.append(lightcurve_file)
                Path(output_areas_file).write_text("areas")
                Path(output_lc_file).write_text("1.0\n2.0\n")
                return {}

            with patch("requests.get", fake_get), patch.object(
                am, "run_convexinv", fake_run_convexinv
            ), patch.object(am, "run_minkowski", lambda *args, **kwargs: (np.zeros((1, 3)), [[1]])):
                am.run_pipeline(
                    lightcurve=url,
                    output_dir=str(tmp_path),
                    asteroid_name="A7753",
                    inversion_options={},
                    conjgradinv_options={},
                )

            expected_download = str(tmp_path / "A7753.lc.txt")
            self.assertEqual(downloaded_files, [expected_download])
            self.assertEqual(used_lightcurve_files, [expected_download])

    def test_download_lightcurve_url_reports_ssl_certificate_failure(self):
        url = "https://example.com/lightcurve.txt"
        ssl_error = requests.exceptions.SSLError("unable to get local issuer certificate")

        with tempfile.TemporaryDirectory() as tmp, patch("requests.get", side_effect=ssl_error):
            with self.assertRaises(am.AsteroidModelError) as caught:
                am._download_lightcurve_url(url, tmp)

        message = str(caught.exception)
        self.assertIn("SSL certificate verification failed", message)
        self.assertIn("pip install certifi", message)

    def test_download_lightcurve_url_retries_damit_certificate_failure(self):
        url = "https://damit.cuni.cz/projects/damit/LightCurves/exportAllForAsteroid/7753/plaintext/A7753.lc.txt"
        ssl_error = requests.exceptions.SSLError("unable to get local issuer certificate")
        calls = []

        def fake_get(source_url, timeout=None, verify=True):
            calls.append((source_url, timeout, verify))
            if len(calls) == 1:
                raise ssl_error
            return FakeRequestsResponse(NATIVE_LIGHTCURVE)

        with tempfile.TemporaryDirectory() as tmp, patch("requests.get", fake_get), self.assertWarns(RuntimeWarning):
            downloaded_file = am._download_lightcurve_url(url, tmp)
            self.assertEqual(Path(downloaded_file).read_text(), NATIVE_LIGHTCURVE)

        self.assertEqual(calls[0], (url, 30, True))
        self.assertEqual(calls[1], (url, 30, False))

    def test_plot_lightcurves_results_passes_period_and_first_observation_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lightcurve_file = tmp_path / "A7753.lc.txt"
            lightcurve_file.write_text(NATIVE_LIGHTCURVE)
            output_dir = tmp_path / "out"
            output_dir.mkdir()
            (output_dir / "A7753_lc_output.csv").write_text("1.0\n2.0\n")
            calls = []

            def fake_plot_lightcurves(*args, **kwargs):
                calls.append((args, kwargs))

            modeler = am.AsteroidModeler("A7753", output_dir=str(output_dir))
            modeler.load_lightcurves(str(lightcurve_file))
            modeler.fit_result = {"period": 33.0}

            with patch.object(am, "plot_lightcurves", fake_plot_lightcurves):
                modeler.plot_lightcurves_results(save=True, show=False)

        self.assertEqual(calls[0][1]["period_hours"], 33.0)
        self.assertEqual(calls[0][1]["zero_time"], 2456877.703423)


class ExampleScriptTests(unittest.TestCase):
    def test_a7753_example_adds_src_to_path_before_importing_pymit(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "docs/examples/model_a7753_from_damit.py"
        source = script.read_text()

        self.assertLess(source.index("sys.path.insert"), source.index("import pymit"))

    def test_a7753_example_imports_without_pythonpath(self):
        repo_root = Path(__file__).resolve().parents[1]
        env = {
            key: value
            for key, value in os.environ.items()
            if key != "PYTHONPATH"
        }
        env["MPLCONFIGDIR"] = "/private/tmp/mplconfig"

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import runpy; runpy.run_path('docs/examples/model_a7753_from_damit.py')",
            ],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
