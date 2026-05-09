import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pymit.lightcurves import (
    _phase_angle_degrees,
    csv_to_lcs_format,
    dataframe_to_lcs_format,
    plot_lightcurves,
)


def _sample_lightcurve_frame():
    return pd.DataFrame(
        [
            {
                "curve_id": "a",
                "is_relative": 0,
                "jd": 2450000.0,
                "brightness": 1.2,
                "sun_x": 1.0,
                "sun_y": 0.0,
                "sun_z": 0.0,
                "earth_x": 0.0,
                "earth_y": 1.0,
                "earth_z": 0.0,
            },
            {
                "curve_id": "a",
                "is_relative": 0,
                "jd": 2450000.1,
                "brightness": 1.3,
                "sun_x": 1.0,
                "sun_y": 0.0,
                "sun_z": 0.0,
                "earth_x": 0.0,
                "earth_y": 1.0,
                "earth_z": 0.0,
            },
            {
                "curve_id": "b",
                "is_relative": 1,
                "jd": 2450001.0,
                "brightness": 0.9,
                "sun_x": 0.0,
                "sun_y": 1.0,
                "sun_z": 0.0,
                "earth_x": 1.0,
                "earth_y": 0.0,
                "earth_z": 0.0,
            },
        ]
    )


class LightcurveConversionTests(unittest.TestCase):
    def test_dataframe_to_lcs_format_groups_curves_and_maps_relative_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_file = Path(tmp) / "input_lcs.txt"

            dataframe_to_lcs_format(_sample_lightcurve_frame(), str(output_file))
            lines = output_file.read_text().splitlines()

        self.assertEqual(lines[0], "2")
        self.assertEqual(lines[1], "2 1")
        self.assertTrue(lines[2].startswith("2450000.000000 1.200000e+00"))
        self.assertEqual(lines[4], "1 0")

    def test_csv_to_lcs_format_matches_dataframe_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_file = Path(tmp) / "input.csv"
            dataframe_output = Path(tmp) / "from_df.txt"
            csv_output = Path(tmp) / "from_csv.txt"
            frame = _sample_lightcurve_frame()
            frame.to_csv(csv_file, index=False)

            dataframe_to_lcs_format(frame, str(dataframe_output))
            csv_to_lcs_format(str(csv_file), str(csv_output))

            self.assertEqual(csv_output.read_text(), dataframe_output.read_text())

    def test_dataframe_to_lcs_format_requires_geometry_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Missing required column"):
                dataframe_to_lcs_format(
                    pd.DataFrame([{"jd": 2450000.0, "brightness": 1.0}]),
                    str(Path(tmp) / "bad.txt"),
                )


class LightcurvePlotTests(unittest.TestCase):
    def test_phase_angle_degrees_uses_sun_and_observer_vectors(self):
        self.assertAlmostEqual(
            _phase_angle_degrees((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            90.0,
        )
        self.assertAlmostEqual(
            _phase_angle_degrees((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            0.0,
        )

    def test_plot_lightcurves_uses_phase_angle_axis_when_vectors_are_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_file = Path(tmp) / "input.lc.txt"
            output_file = Path(tmp) / "modeled.txt"
            input_file.write_text(
                "1\n"
                "2 0\n"
                "2450000.0 1.0 1 0 0 0 1 0\n"
                "2450001.0 1.2 1 0 0 1 0 0\n"
            )
            output_file.write_text("1.1\n1.3\n")

            with (
                patch("pymit.lightcurves.plt.plot") as plot_mock,
                patch("pymit.lightcurves.plt.xlabel") as xlabel_mock,
                patch("pymit.lightcurves.plt.figure"),
                patch("pymit.lightcurves.plt.legend"),
                patch("pymit.lightcurves.plt.show"),
                patch("pymit.lightcurves.plt.close"),
            ):
                plot_lightcurves(str(input_file), str(output_file), show=False)

        self.assertEqual(plot_mock.call_args_list[0].args[0], [90.0, 0.0])
        xlabel_mock.assert_called_once_with("Solar Phase Angle (deg)")


if __name__ == "__main__":
    unittest.main()
