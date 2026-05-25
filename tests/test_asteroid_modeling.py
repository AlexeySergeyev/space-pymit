import contextlib
import tempfile
import unittest
import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import requests

import pymit.asteroid_modeling as am
from pymit.errors import AsteroidModelError


NATIVE_LIGHTCURVE = """1
2 0
2456877.703423 6.364793e+00 2.892962e+00 7.633510e-01 -5.152975e-02 3.620724e+00 4.246431e-02 -5.157907e-02
2456878.027575 4.488047e+00 2.891985e+00 7.665565e-01 -5.141167e-02 3.623596e+00 4.965999e-02 -5.144869e-02
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

    def test_run_inversion_defaults_zero_time_to_rounded_first_observation_jd(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lightcurve_file = tmp_path / "A7753.lc.txt"
            lightcurve_file.write_text(NATIVE_LIGHTCURVE)
            zero_time_values = []

            def fake_run_convexinv(param_file, lightcurve_file, output_areas_file, output_lc_file, verbose=False):
                zero_time_values.append(Path(param_file).read_text().splitlines()[3].split()[0])
                Path(output_areas_file).write_text("areas")
                Path(output_lc_file).write_text("1.0\n2.0\n")
                return {"period": 1.23}

            with patch.object(am, "run_convexinv", fake_run_convexinv), patch.object(
                am, "run_minkowski", lambda *args, **kwargs: (np.zeros((1, 3)), [[1]])
            ):
                modeler = am.AsteroidModeler(asteroid_name="A7753", output_dir=str(tmp_path / "out"))
                modeler.load_lightcurves(str(lightcurve_file)).run_inversion()

        self.assertEqual(zero_time_values, ["2456878"])

    def test_run_inversion_preserves_explicit_zero_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lightcurve_file = tmp_path / "A7753.lc.txt"
            lightcurve_file.write_text(NATIVE_LIGHTCURVE)
            zero_time_values = []

            def fake_run_convexinv(param_file, lightcurve_file, output_areas_file, output_lc_file, verbose=False):
                zero_time_values.append(Path(param_file).read_text().splitlines()[3].split()[0])
                Path(output_areas_file).write_text("areas")
                Path(output_lc_file).write_text("1.0\n2.0\n")
                return {"period": 1.23}

            with patch.object(am, "run_convexinv", fake_run_convexinv), patch.object(
                am, "run_minkowski", lambda *args, **kwargs: (np.zeros((1, 3)), [[1]])
            ):
                modeler = am.AsteroidModeler(asteroid_name="A7753", output_dir=str(tmp_path / "out"))
                (
                    modeler.load_lightcurves(str(lightcurve_file))
                    .load_parameters(inversion_json={"zero_time": 2456000.5})
                    .run_inversion()
                )

        self.assertEqual(zero_time_values, ["2456000.5"])

    def test_run_inversion_prints_convexinv_summary_before_minkowski(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lightcurve_file = tmp_path / "A7753.lc.txt"
            lightcurve_file.write_text(NATIVE_LIGHTCURVE)

            def fake_run_convexinv(param_file, lightcurve_file, output_areas_file, output_lc_file, verbose=False):
                Path(output_areas_file).write_text("areas")
                Path(output_lc_file).write_text("1.0\n2.0\n")
                return {
                    "chi_square": 1.341417,
                    "dev": 0.034701,
                    "lambda": -8.394525,
                    "beta": -4.563390,
                    "period": 4.260875,
                    "phase_a": 0.5,
                    "phase_d": 0.1,
                    "phase_k": -0.5,
                    "lambert_c": 0.1,
                    "shadow_percent": 1.08,
                }

            stdout = io.StringIO()
            with patch.object(am, "run_convexinv", fake_run_convexinv), patch.object(
                am, "run_minkowski", lambda *args, **kwargs: (np.zeros((1, 3)), [[1]])
            ), contextlib.redirect_stdout(stdout):
                modeler = am.AsteroidModeler(asteroid_name="A7753", output_dir=str(tmp_path / "out"))
                modeler.load_lightcurves(str(lightcurve_file)).run_inversion(verbose=True)

            output = stdout.getvalue()

        summary_index = output.index("convexinv fit summary:")
        minkowski_index = output.index("Reconstructing 3D shape from generated features using minkowski...")
        self.assertLess(summary_index, minkowski_index)
        self.assertIn("chi-square: 1.341417", output)
        self.assertIn("lambda/beta/period: -8.394525 / -4.56339 / 4.260875 h", output)
        self.assertIn("phase a/d/k: 0.5 / 0.1 / -0.5", output)
        self.assertIn("Lambert coefficient c: 0.1", output)
        self.assertIn("dark facet area: 1.08%", output)

    def test_convex_and_minkowski_can_run_as_separate_logged_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lightcurve_file = tmp_path / "A7753.lc.txt"
            lightcurve_file.write_text(NATIVE_LIGHTCURVE)
            stage_events = []

            def fake_run_convexinv(param_file, lightcurve_file, output_areas_file, output_lc_file, verbose=False):
                stage_events.append("convex")
                Path(output_areas_file).write_text("areas")
                Path(output_lc_file).write_text("1.0\n2.0\n")
                return {"period": 1.23, "chi_square": 4.56}

            def fake_run_minkowski(areas_file, pwd_dir=None, verbose=False):
                stage_events.append(("minkowski", Path(areas_file).name))
                return np.zeros((1, 3)), [[1]]

            stdout = io.StringIO()
            with patch.object(am, "run_convexinv", fake_run_convexinv), patch.object(
                am, "run_minkowski", fake_run_minkowski
            ), contextlib.redirect_stdout(stdout):
                modeler = am.AsteroidModeler(asteroid_name="A7753", output_dir=str(tmp_path / "out"))
                modeler.load_lightcurves(str(lightcurve_file))
                fit_result = modeler.run_convex_inversion(verbose=True)
                vertices, faces = modeler.run_minkowski_reconstruction(verbose=True)

            output = stdout.getvalue()

        self.assertEqual(fit_result, {"period": 1.23, "chi_square": 4.56})
        self.assertEqual(vertices.shape, (1, 3))
        self.assertEqual(faces, [[1]])
        self.assertEqual(stage_events, ["convex", ("minkowski", "A7753_areas.txt")])
        self.assertLess(
            output.index("convexinv fit summary:"),
            output.index("Reconstructing 3D shape from generated features using minkowski..."),
        )

    def test_run_inversion_can_scan_period_before_convexinv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lightcurve_file = tmp_path / "A7753.lc.txt"
            lightcurve_file.write_text(NATIVE_LIGHTCURVE)
            output_dir = tmp_path / "out"
            convex_periods = []

            def fake_run_period_scan(
                param_file,
                lightcurve_file,
                output_periods_file,
                verbose=False,
                stdout_log_file=None,
            ):
                Path(output_periods_file).write_text(
                    "5.750000 0.400000 40.000000 12 1.0\n"
                    "5.760000 0.200000 20.000000 13 2.0\n"
                )
                if stdout_log_file:
                    Path(stdout_log_file).write_text("stdout")

            def fake_run_convexinv(param_file, lightcurve_file, output_areas_file, output_lc_file, verbose=False):
                convex_periods.append(Path(param_file).read_text().splitlines()[2].split()[0])
                Path(output_areas_file).write_text("areas")
                Path(output_lc_file).write_text("1.0\n2.0\n")
                return {"period": 5.761, "chi_square": 19.0}

            with patch.object(am, "run_period_scan", fake_run_period_scan), patch.object(
                am, "run_convexinv", fake_run_convexinv
            ), patch.object(am, "run_minkowski", lambda *args, **kwargs: (np.zeros((1, 3)), [[1]])):
                modeler = am.AsteroidModeler("A7753", output_dir=str(output_dir))
                modeler.load_lightcurves(str(lightcurve_file)).run_inversion(
                    run_period_scan=True,
                    period_scan_options={
                        "period_start": 5.7,
                        "period_end": 5.9,
                    },
                )

            raw_output = output_dir / "A7753_period_scan.txt"
            csv_output = output_dir / "A7753_period_scan.csv"
            plot_output = output_dir / "A7753_period_scan.png"

            self.assertEqual(convex_periods, ["5.76"])
            self.assertEqual(modeler.period_scan_result.period_hours, 5.76)
            self.assertEqual(modeler.fit_result["initial_period_from_period_scan"], 5.76)
            self.assertTrue(raw_output.exists())
            self.assertIn("5.760000", raw_output.read_text())
            self.assertTrue(csv_output.exists())
            self.assertEqual(plot_output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_run_period_scan_splits_period_range_across_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lightcurve_file = tmp_path / "A7753.lc.txt"
            lightcurve_file.write_text(NATIVE_LIGHTCURVE)
            output_dir = tmp_path / "out"
            scanned_ranges = []

            def fake_run_period_scan(
                param_file,
                lightcurve_file,
                output_periods_file,
                verbose=False,
                stdout_log_file=None,
            ):
                period_start, period_end, _coefficient = [
                    float(value)
                    for value in Path(param_file).read_text().splitlines()[0].split()[:3]
                ]
                scanned_ranges.append((period_start, period_end))
                chi_square = 30.0 if period_end <= 6.0 else 10.0
                Path(output_periods_file).write_text(
                    f"{period_end:.6f} 0.200000 {chi_square:.6f} 13 2.0\n"
                )
                if stdout_log_file:
                    Path(stdout_log_file).write_text("stdout")

            with patch.object(am, "run_period_scan", fake_run_period_scan):
                modeler = am.AsteroidModeler("A7753", output_dir=str(output_dir))
                best = modeler.load_lightcurves(str(lightcurve_file)).run_period_scan(
                    period_scan_options={
                        "period_start": 5.0,
                        "period_end": 7.0,
                    },
                    workers=2,
                )

            merged_output = output_dir / "A7753_period_scan.txt"
            csv_output = output_dir / "A7753_period_scan.csv"

            self.assertEqual(sorted(scanned_ranges), [(5.0, 6.0), (6.0, 7.0)])
            self.assertEqual(best.period_hours, 7.0)
            self.assertEqual(modeler.inversion_options["initial_period"], 7.0)
            self.assertEqual(len(merged_output.read_text().splitlines()), 2)
            self.assertTrue(csv_output.exists())

    def test_pole_grid_scan_selects_minimum_chi_square_and_reconstructs_best_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lightcurve_file = tmp_path / "A7753.lc.txt"
            lightcurve_file.write_text(NATIVE_LIGHTCURVE)
            output_dir = tmp_path / "out"
            candidate_params = []
            reconstructed_areas = []

            def fake_run_convexinv(
                param_file,
                lightcurve_file,
                output_areas_file,
                output_lc_file,
                verbose=False,
                stdout_log_file=None,
            ):
                lines = Path(param_file).read_text().splitlines()
                initial_lambda = float(lines[0].split()[0])
                initial_beta = float(lines[1].split()[0])
                zero_time = lines[3].split()[0]
                candidate_params.append((initial_lambda, initial_beta))
                self.assertEqual(zero_time, "2456878")
                if initial_lambda > 100.0 and initial_lambda < 200.0:
                    raise AsteroidModelError("convexinv failed")

                Path(output_areas_file).write_text(
                    f"1\n{initial_lambda:.3f}\n0.0 0.0 1.0\n"
                )
                Path(output_lc_file).write_text(f"{initial_lambda:.3f}\n")
                if stdout_log_file:
                    Path(stdout_log_file).write_text("stdout")

                chi_square = 10.0 if initial_lambda == 0.0 else 3.0
                return {
                    "lambda": initial_lambda + 1.0,
                    "beta": initial_beta + 1.0,
                    "period": 33.0,
                    "chi_square": chi_square,
                    "dev": 0.1,
                    "shadow_percent": 0.5,
                }

            def fake_run_minkowski(areas_file, pwd_dir=None, verbose=False):
                reconstructed_areas.append(Path(areas_file).read_text())
                return np.zeros((1, 3)), [[1]]

            with patch.object(am, "run_convexinv", fake_run_convexinv), patch.object(
                am, "run_minkowski", fake_run_minkowski
            ):
                modeler = am.AsteroidModeler("A7753", output_dir=str(output_dir))
                vertices, faces = (
                    modeler.load_lightcurves(str(lightcurve_file))
                    .load_parameters(
                        inversion_json={
                            "initial_period": 33.0,
                            "initial_lambda_fixed": 1,
                            "initial_beta_fixed": 1,
                        }
                    )
                    .run_pole_grid_scan(n=1, workers=1, verbose=False)
                )

            results_csv = output_dir / "A7753_pole_scan_results.csv"
            best_json = output_dir / "A7753_pole_scan_best.json"
            map_png = output_dir / "A7753_pole_scan_map.png"
            fitted_map_png = output_dir / "A7753_pole_scan_map_fitted.png"
            with results_csv.open() as f:
                rows = list(csv.DictReader(f))
            best = json.loads(best_json.read_text())
            standard_areas_text = (output_dir / "A7753_areas.txt").read_text()
            map_header = map_png.read_bytes()[:8]
            fitted_map_header = fitted_map_png.read_bytes()[:8]

        self.assertEqual(len(candidate_params), 3)
        self.assertEqual(vertices.shape, (1, 3))
        self.assertEqual(faces, [[1]])
        self.assertEqual([row["status"] for row in rows], ["failed", "success", "success"])
        self.assertEqual(best["index"], 2)
        self.assertEqual(best["chi_square"], 3.0)
        self.assertEqual(modeler.fit_result["chi_square"], 3.0)
        self.assertEqual(standard_areas_text, reconstructed_areas[0])
        self.assertIn("222.492", standard_areas_text)
        self.assertEqual(map_header, b"\x89PNG\r\n\x1a\n")
        self.assertEqual(fitted_map_header, b"\x89PNG\r\n\x1a\n")

    def test_pole_grid_scan_skips_lowest_chi_square_when_reconstruction_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lightcurve_file = tmp_path / "A7753.lc.txt"
            lightcurve_file.write_text(NATIVE_LIGHTCURVE)
            output_dir = tmp_path / "out"
            reconstructed_areas = []

            def fake_run_convexinv(
                param_file,
                lightcurve_file,
                output_areas_file,
                output_lc_file,
                verbose=False,
                stdout_log_file=None,
            ):
                initial_lambda = float(Path(param_file).read_text().splitlines()[0].split()[0])
                Path(output_areas_file).write_text(
                    f"1\n{initial_lambda:.3f}\n0.0 0.0 1.0\n"
                )
                Path(output_lc_file).write_text(f"{initial_lambda:.3f}\n")
                if stdout_log_file:
                    Path(stdout_log_file).write_text("stdout")
                return {
                    "lambda": initial_lambda,
                    "beta": 0.0,
                    "period": 33.0,
                    "chi_square": 1.0 if initial_lambda == 0.0 else initial_lambda,
                    "dev": 0.1,
                    "shadow_percent": 0.5,
                }

            def fake_run_minkowski(areas_file, pwd_dir=None, verbose=False):
                areas_text = Path(areas_file).read_text()
                if "\n0.000\n" in areas_text:
                    raise AsteroidModelError("minkowski crashed")
                reconstructed_areas.append(areas_text)
                return np.zeros((1, 3)), [[1]]

            with patch.object(am, "run_convexinv", fake_run_convexinv), patch.object(
                am, "run_minkowski", fake_run_minkowski
            ):
                modeler = am.AsteroidModeler("A7753", output_dir=str(output_dir))
                vertices, faces = (
                    modeler.load_lightcurves(str(lightcurve_file))
                    .load_parameters(inversion_json={"initial_period": 33.0})
                    .run_pole_grid_scan(n=1, workers=1, verbose=False)
                )

            with (output_dir / "A7753_pole_scan_results.csv").open() as f:
                rows = list(csv.DictReader(f))
            best = json.loads((output_dir / "A7753_pole_scan_best.json").read_text())
            standard_areas_text = (output_dir / "A7753_areas.txt").read_text()

        self.assertEqual(vertices.shape, (1, 3))
        self.assertEqual(faces, [[1]])
        self.assertEqual(rows[1]["status"], "reconstruction_failed")
        self.assertIn("minkowski crashed", rows[1]["error"])
        self.assertEqual(best["index"], 0)
        self.assertEqual(modeler.fit_result["pole_scan_index"], 0)
        self.assertEqual(standard_areas_text, reconstructed_areas[0])
        self.assertIn("137.508", standard_areas_text)

    def test_modeler_repairs_malformed_native_lightcurve_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lightcurve_file = tmp_path / "lcs4DAMIT_lite.txt"
            lightcurve_file.write_text(
                "2\n"
                "1 0\n"
                "2450000.0 1.0 1 0 0 0 1 0\n"
                "2 0\n"
                "2450001.0 0.9 1 0 0 0 1 0\n"
            )
            used_lightcurve_files = []

            def fake_run_convexinv(param_file, lightcurve_file, output_areas_file, output_lc_file, verbose=False):
                used_lightcurve_files.append(lightcurve_file)
                Path(output_areas_file).write_text("areas")
                Path(output_lc_file).write_text("1.0\n1.0\n")
                return {"period": 4.0}

            with patch.object(am, "run_convexinv", fake_run_convexinv), patch.object(
                am, "run_minkowski", lambda *args, **kwargs: (np.zeros((1, 3)), [[1]])
            ):
                modeler = am.AsteroidModeler(asteroid_name="Bad", output_dir=str(tmp_path / "out"))
                modeler.load_lightcurves(str(lightcurve_file)).run_inversion()

            repaired_file = tmp_path / "out" / "lcs4DAMIT_lite_convexinv.txt"
            self.assertEqual(used_lightcurve_files, [str(repaired_file)])
            self.assertEqual(repaired_file.read_text().splitlines()[3], "1 0")

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
        self.assertEqual(
            calls[0][1]["save_path"],
            str(output_dir / "A7753_lightcurves.html"),
        )


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
