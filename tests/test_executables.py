import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pymit.errors import AsteroidModelError
from pymit.executables import (
    _parse_convexinv_output,
    run_convexinv,
    run_minkowski,
    run_period_scan,
)


VALID_AREAS = """1
1.0
0.0 0.0 1.0
"""


class MinkowskiExecutionTests(unittest.TestCase):
    def test_run_minkowski_reports_sigbus_as_degenerate_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            areas_file = Path(tmp) / "areas.txt"
            areas_file.write_text(VALID_AREAS)

            proc = Mock()
            proc.communicate.return_value = (
                "",
                "Program received signal SIGBUS: Access to an undefined portion of a memory object.",
            )
            proc.returncode = -10

            with patch("pymit.executables.subprocess.Popen", return_value=proc):
                with self.assertRaises(AsteroidModelError) as caught:
                    run_minkowski(str(areas_file), pwd_dir=tmp)

        message = str(caught.exception)
        self.assertIn("minkowski crashed with signal SIGBUS", message)
        self.assertIn("areas/normals file may be invalid or numerically degenerate", message)
        self.assertIn("areas.txt", message)

    def test_run_minkowski_rejects_malformed_areas_file_before_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            areas_file = Path(tmp) / "bad_areas.txt"
            areas_file.write_text("2\n1.0\n0.0 0.0 1.0\n")

            with patch("pymit.executables.subprocess.Popen") as popen:
                with self.assertRaises(AsteroidModelError) as caught:
                    run_minkowski(str(areas_file), pwd_dir=tmp)

        popen.assert_not_called()
        self.assertIn("Malformed minkowski input", str(caught.exception))
        self.assertIn("expected 5 non-empty lines", str(caught.exception))

    def test_run_minkowski_rejects_extreme_area_spread_before_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            areas_file = Path(tmp) / "degenerate_areas.txt"
            areas_file.write_text(
                "2\n"
                "1.0e-6\n"
                "0.0 0.0 1.0\n"
                "2.0\n"
                "0.0 1.0 0.0\n"
            )

            with patch("pymit.executables.subprocess.Popen") as popen:
                with self.assertRaises(AsteroidModelError) as caught:
                    run_minkowski(str(areas_file), pwd_dir=tmp)

        popen.assert_not_called()
        self.assertIn("numerically degenerate", str(caught.exception))
        self.assertIn("area max/min ratio", str(caught.exception))

    def test_run_minkowski_times_out_and_kills_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            areas_file = Path(tmp) / "areas.txt"
            areas_file.write_text(VALID_AREAS)

            proc = Mock()
            proc.communicate.side_effect = [
                subprocess.TimeoutExpired(cmd=["minkowski"], timeout=0.1),
                ("", ""),
            ]

            with patch("pymit.executables.subprocess.Popen", return_value=proc):
                with self.assertRaises(AsteroidModelError) as caught:
                    run_minkowski(str(areas_file), pwd_dir=tmp, timeout_seconds=0.1)

        proc.communicate.assert_any_call(timeout=0.1)
        proc.kill.assert_called_once()
        message = str(caught.exception)
        self.assertIn("minkowski timed out after 0.1 seconds", message)
        self.assertIn("areas.txt", message)


class ConvexInvExecutionTests(unittest.TestCase):
    def test_parse_convexinv_output_extracts_fit_metrics_and_dark_facet(self):
        output = (
            "1  chi2 120.500000  dev 0.310000  alambda 0.001000\n"
            "2  chi2 99.250000  dev 0.280000  alambda 0.000100\n"
            "\n"
            "lambda, beta and period (hrs): 123.400000 -56.700000 33.000000\n"
            "phase function parameters: 0.5 0.1 -1.05\n"
            "Lambert coefficient: 0.1\n"
            "plus a dark facet with area 0.42%\n"
        )

        fit = _parse_convexinv_output(output)

        self.assertEqual(fit["lambda"], 123.4)
        self.assertEqual(fit["beta"], -56.7)
        self.assertEqual(fit["period"], 33.0)
        self.assertEqual(fit["chi_square"], 99.25)
        self.assertEqual(fit["dev"], 0.28)
        self.assertEqual(fit["shadow_percent"], 0.42)
        self.assertEqual(fit["phase_a"], 0.5)
        self.assertEqual(fit["phase_d"], 0.1)
        self.assertEqual(fit["phase_k"], -1.05)
        self.assertEqual(fit["lambert_c"], 0.1)

    def test_run_convexinv_reports_singular_matrix_with_actual_free_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            param_file = tmp_path / "input_convexinv"
            lightcurve_file = tmp_path / "input_lcs.txt"
            areas_file = tmp_path / "areas.txt"
            output_lc_file = tmp_path / "output_lc.csv"
            param_file.write_text(
                "180.0\t1\tinital lambda [deg] (0/1 - fixed/free)\n"
                "-60.0\t1\tinitial beta [deg] (0/1 - fixed/free)\n"
                "4.26\t1\tinital period [hours] (0/1 - fixed/free)\n"
                "2455483\t\tzero time [JD]\n"
                "0\t\tinitial rotation angle [deg]\n"
                "0.1\t\tconvexity regularization\n"
                "4 4\t\tdegree and order of spherical harmonics expansion\n"
                "6\t\tnumber of rows\n"
                "0.5\t1\tphase funct. param. 'a' (0/1 - fixed/free)\n"
                "0.1\t1\tphase funct. param. 'd' (0/1 - fixed/free)\n"
                "-0.5\t0\tphase funct. param. 'k' (0/1 - fixed/free)\n"
                "0.1\t0\tLambert coefficient 'c' (0/1 - fixed/free)\n"
                "50\t\titeration stop condition\n"
            )
            lightcurve_file.write_text("1\n1 1\n2450000.0 1.0 0 0 1 0 1 0\n")

            proc = Mock()
            proc.stdout = ["gaussj: Singular Matrix-2.\n"]
            proc.wait.return_value = None
            proc.returncode = 3

            with patch("pymit.executables.subprocess.Popen", return_value=proc):
                with self.assertRaises(AsteroidModelError) as caught:
                    run_convexinv(
                        str(param_file),
                        str(lightcurve_file),
                        str(areas_file),
                        str(output_lc_file),
                    )

        message = str(caught.exception)
        self.assertIn("convexinv failed because its linear solve became singular", message)
        self.assertIn("Free parameters in", message)
        self.assertIn("initial lambda", message)
        self.assertIn("initial beta", message)
        self.assertIn("initial period", message)
        self.assertIn("phase function a / DAMIT p2", message)
        self.assertIn("phase function d / DAMIT p3", message)
        self.assertIn("Lambert coefficient c is fixed", message)
        self.assertNotIn("For A7753", message)

    def test_run_convexinv_merges_stderr_into_stdout_to_avoid_pipe_deadlock(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            param_file = tmp_path / "input_convexinv"
            lightcurve_file = tmp_path / "input_lcs.txt"
            areas_file = tmp_path / "areas.txt"
            output_lc_file = tmp_path / "output_lc.csv"
            param_file.write_text("params\n")
            lightcurve_file.write_text("1\n1 1\n2450000.0 1.0 0 0 1 0 1 0\n")

            proc = Mock()
            proc.stdout = ["lambda, beta and period : 1 2 3\n"]
            proc.wait.return_value = None
            proc.returncode = 0

            with patch("pymit.executables.subprocess.Popen", return_value=proc) as popen:
                run_convexinv(
                    str(param_file),
                    str(lightcurve_file),
                    str(areas_file),
                    str(output_lc_file),
                )

        _, kwargs = popen.call_args
        self.assertEqual(kwargs["stderr"], subprocess.STDOUT)


class PeriodScanExecutionTests(unittest.TestCase):
    def test_run_period_scan_streams_lightcurves_to_period_scan_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            param_file = tmp_path / "input_period_scan.txt"
            lightcurve_file = tmp_path / "input_lcs.txt"
            output_file = tmp_path / "periods.txt"
            param_file.write_text("params\n")
            lightcurve_file.write_text("1\n1 1\n2450000.0 1.0 0 0 1 0 1 0\n")

            proc = Mock()
            proc.stdout = ["period   rms      chi2      iter. dark area %\n"]
            proc.wait.return_value = None
            proc.returncode = 0

            with patch("pymit.executables.subprocess.Popen", return_value=proc) as popen:
                run_period_scan(str(param_file), str(lightcurve_file), str(output_file))

        cmd, kwargs = popen.call_args
        self.assertEqual(cmd[0][1:], [str(param_file), str(output_file)])
        self.assertEqual(kwargs["stderr"], subprocess.STDOUT)


if __name__ == "__main__":
    unittest.main()
