import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pymit.errors import AsteroidModelError
from pymit.executables import run_convexinv, run_minkowski


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


class ConvexInvExecutionTests(unittest.TestCase):
    def test_run_convexinv_reports_singular_matrix_with_lambert_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            param_file = tmp_path / "input_convexinv"
            lightcurve_file = tmp_path / "input_lcs.txt"
            areas_file = tmp_path / "areas.txt"
            output_lc_file = tmp_path / "output_lc.csv"
            param_file.write_text("0.1\t1\tLambert coefficient 'c' (0/1 - fixed/free)\n")
            lightcurve_file.write_text("1\n1 1\n2450000.0 1.0 0 0 1 0 1 0\n")

            proc = Mock()
            proc.stdout = []
            proc.stderr.read.return_value = "gaussj: Singular Matrix-2.\n"
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
        self.assertIn("keep the Lambert coefficient fixed", message)
        self.assertIn("phase_func_c_fixed=0", message)


if __name__ == "__main__":
    unittest.main()
