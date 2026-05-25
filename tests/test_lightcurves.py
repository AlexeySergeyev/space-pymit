import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pymit.lightcurves import (
    _phase_angle_degrees,
    build_lightcurve_figure,
    csv_to_lcs_format,
    dataframe_to_lcs_format,
    normalize_native_lcs_format,
    plot_lightcurves,
)
from pymit.parameters import create_period_scan_param_file
from pymit.period_scan import (
    find_best_period_scan_result,
    parse_period_scan_output,
    save_period_scan_plot,
    write_period_scan_csv,
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
            with self.assertRaisesRegex(
                ValueError, "DataFrame lightcurve input is missing required columns"
            ):
                dataframe_to_lcs_format(
                    pd.DataFrame([{"jd": 2450000.0, "brightness": 1.0}]),
                    str(Path(tmp) / "bad.txt"),
                )

    def test_normalize_native_lcs_format_repairs_overstated_block_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_file = Path(tmp) / "lcs4damit_lite.txt"
            output_file = Path(tmp) / "normalized.lc.txt"
            input_file.write_text(
                "2\n"
                "2 0\n"
                "2450000.0 1.0 1 0 0 0 1 0\n"
                "2450000.1 1.1 1 0 0 0 1 0\n"
                "3 0\n"
                "2450001.0 0.9 1 0 0 0 1 0\n"
                "2450001.1 1.0 1 0 0 0 1 0\n"
            )

            changed = normalize_native_lcs_format(input_file, output_file)

            lines = output_file.read_text().splitlines()

        self.assertTrue(changed)
        self.assertEqual(lines[0], "2")
        self.assertEqual(lines[1], "2 0")
        self.assertEqual(lines[4], "2 0")
        self.assertEqual(len(lines), 7)

    def test_normalize_native_lcs_format_repairs_understated_last_block_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_file = Path(tmp) / "lcs4damit_lite.txt"
            output_file = Path(tmp) / "normalized.lc.txt"
            input_file.write_text(
                "1\n"
                "1 0\n"
                "2450000.0 1.0 1 0 0 0 1 0\n"
                "2450000.1 1.1 1 0 0 0 1 0\n"
            )

            changed = normalize_native_lcs_format(input_file, output_file)

            lines = output_file.read_text().splitlines()

        self.assertTrue(changed)
        self.assertEqual(lines[0], "1")
        self.assertEqual(lines[1], "2 0")
        self.assertEqual(len(lines), 4)

    def test_normalize_native_lcs_format_preserves_valid_native_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_file = Path(tmp) / "valid.lc.txt"
            output_file = Path(tmp) / "normalized.lc.txt"
            input_file.write_text(
                "1\n"
                "2 0\n"
                "2450000.000000 1.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n"
                "2450000.100000 1.100000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n"
            )

            changed = normalize_native_lcs_format(input_file, output_file)

            lines = output_file.read_text().splitlines()

        self.assertFalse(changed)
        self.assertEqual(lines[0], "1")
        self.assertEqual(lines[1], "2 0")
        self.assertEqual(len(lines), 4)

    def test_normalize_native_lcs_format_writes_canonical_damit_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_file = Path(tmp) / "raw.lc.txt"
            output_file = Path(tmp) / "normalized.lc.txt"
            input_file.write_text(
                "1\n"
                "1 0\n"
                "2455482.5757692647 1.1045867154760998 -1.007702391236135 "
                "-0.253689143758588 -0.03464203452935782 -0.06694335961248796 "
                "0.07904960996216315 -0.03461882825716007 \n"
            )

            normalize_native_lcs_format(input_file, output_file)

            lines = output_file.read_text().splitlines()

        self.assertEqual(
            lines[2],
            "2455482.575769 1.104587e+00 -1.007702e+00 -2.536891e-01 "
            "-3.464203e-02 -6.694336e-02 7.904961e-02 -3.461883e-02",
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

    def test_build_lightcurve_figure_uses_phase_angle_axis_when_vectors_are_available(self):
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

            fig = build_lightcurve_figure(str(input_file), str(output_file))

        self.assertEqual(fig.layout.xaxis.title.text, "Solar Phase Angle (deg)")
        self.assertEqual(list(fig.data[0].x), [90.0, 0.0])
        self.assertEqual(list(fig.data[0].y), [1.0, 1.2])
        self.assertEqual(list(fig.data[1].y), [1.1, 1.3])
        self.assertEqual(fig.data[0].name, "Observed Curve 1")
        self.assertEqual(fig.data[1].name, "Modeled Curve 1")
        self.assertNotRegex(fig.data[0].marker.color, r"^C\d+$")
        self.assertEqual(fig.data[0].marker.color, fig.data[1].marker.color)

    def test_build_lightcurve_figure_limits_displayed_curves(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_file = Path(tmp) / "input.lc.txt"
            output_file = Path(tmp) / "modeled.txt"
            input_file.write_text(
                "2\n"
                "1 0\n"
                "2450000.0 1.0 1 0 0 0 1 0\n"
                "1 0\n"
                "2450001.0 2.0 1 0 0 0 1 0\n"
            )
            output_file.write_text("1.1\n2.1\n")

            fig = build_lightcurve_figure(str(input_file), str(output_file), max_curves=1)

        self.assertEqual(len(fig.data), 2)
        self.assertEqual(fig.data[0].name, "Observed Curve 1")
        self.assertEqual(fig.data[1].name, "Modeled Curve 1")

    def test_plot_lightcurves_writes_interactive_html_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_file = Path(tmp) / "input.lc.txt"
            output_file = Path(tmp) / "modeled.txt"
            html_file = Path(tmp) / "lightcurves.html"
            input_file.write_text(
                "1\n"
                "1 0\n"
                "2450000.0 1.0 1 0 0 0 1 0\n"
            )
            output_file.write_text("1.1\n")

            fig = plot_lightcurves(
                str(input_file), str(output_file), save_path=str(html_file), show=False
            )
            html_text = html_file.read_text().lower()

        self.assertEqual(len(fig.data), 2)
        self.assertIn("<html>", html_text)
        self.assertIn("plotly", html_text)


class PeriodScanTests(unittest.TestCase):
    def test_create_period_scan_param_file_matches_damit_input_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            param_file = Path(tmp) / "input_period_scan.txt"

            create_period_scan_param_file(
                {
                    "period_start": 5.7,
                    "period_end": 5.9,
                    "period_interval_coefficient": 0.6,
                    "number_of_rows": 6,
                },
                str(param_file),
            )

            lines = param_file.read_text().splitlines()

        self.assertEqual(lines[0].split()[:3], ["5.7", "5.9", "0.6"])
        self.assertEqual(lines[3].split()[0], "6")
        self.assertEqual(lines[-2].split()[0], "50")
        self.assertEqual(lines[-1].split()[0], "10")

    def test_create_period_scan_param_file_defaults_to_low_resolution_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            param_file = Path(tmp) / "input_period_scan.txt"

            create_period_scan_param_file({}, str(param_file))

            lines = param_file.read_text().splitlines()

        self.assertEqual(lines[2].split()[:2], ["3", "3"])
        self.assertEqual(lines[3].split()[0], "4")

    def test_create_period_scan_param_file_accepts_custom_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            param_file = Path(tmp) / "input_period_scan.txt"

            create_period_scan_param_file(
                {
                    "spherical_harmonics_degree": 5,
                    "spherical_harmonics_order": 4,
                    "number_of_rows": 7,
                },
                str(param_file),
            )

            lines = param_file.read_text().splitlines()

        self.assertEqual(lines[2].split()[:2], ["5", "4"])
        self.assertEqual(lines[3].split()[0], "7")

    def test_parse_period_scan_output_selects_minimum_chi_square_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_file = tmp_path / "periods.txt"
            csv_file = tmp_path / "periods.csv"
            plot_file = tmp_path / "periods.png"
            output_file.write_text(
                "5.75 0.4 40.0 12 1.0\n"
                "5.76 0.2 20.0 13 2.0\n"
                "5.77 0.3 30.0 14 3.0\n"
            )

            results = parse_period_scan_output(output_file)
            best = find_best_period_scan_result(results)
            write_period_scan_csv(results, csv_file)
            save_period_scan_plot(results, plot_file, best_result=best)

            csv_lines = csv_file.read_text().splitlines()
            plot_header = plot_file.read_bytes()[:8]

        self.assertEqual(best.period_hours, 5.76)
        self.assertEqual(best.chi_square, 20.0)
        self.assertEqual(csv_lines[0], "period_hours,rms,chi_square,iterations,shadow_percent")
        self.assertEqual(plot_header, b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
