import tempfile
import unittest
from pathlib import Path

from pymit.parameters import create_conjgradinv_param_file, create_convexinv_param_file


class ParameterFileTests(unittest.TestCase):
    def test_create_convexinv_param_file_applies_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_file = Path(tmp) / "input_convexinv"

            create_convexinv_param_file(
                {
                    "initial_lambda": 123,
                    "initial_period": 7.5,
                    "number_of_rows": 12,
                    "iteration_stop_condition": 25,
                },
                str(output_file),
            )

            text = output_file.read_text()

        self.assertIn("123\t1\tinital lambda", text)
        self.assertIn("7.5\t1\tinital period", text)
        self.assertIn("12\t\tnumber of rows", text)
        self.assertIn("25\t\titeration stop condition", text)

    def test_create_convexinv_param_file_uses_damit_lsl_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_file = Path(tmp) / "input_convexinv"

            create_convexinv_param_file({}, str(output_file))

            text = output_file.read_text()

        self.assertIn("0.5\t0\tphase funct. param. 'a'", text)
        self.assertIn("0.1\t0\tphase funct. param. 'd'", text)
        self.assertIn("-1.05\t0\tphase funct. param. 'k'", text)
        self.assertIn("0.1\t0\tLambert coefficient 'c'", text)

    def test_create_conjgradinv_param_file_applies_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_file = Path(tmp) / "input_conjgradinv"

            create_conjgradinv_param_file(
                {
                    "convexity_weight": 0.4,
                    "number_of_rows": 10,
                    "number_of_iterations": 250,
                },
                str(output_file),
            )

            lines = output_file.read_text().splitlines()

        self.assertEqual(lines[0], "0.4\t\t\tconvexity weight")
        self.assertEqual(lines[1], "10\t\t\tnumber of rows")
        self.assertEqual(lines[2], "250\t\t\tnumber of iterations")


if __name__ == "__main__":
    unittest.main()
