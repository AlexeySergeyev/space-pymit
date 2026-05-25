import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from pymit.pole_scan import (
    PoleScanCandidateResult,
    build_pole_scan_map_figure,
    golden_spiral_g10,
    save_pole_scan_map_matplotlib,
    write_best_result,
    write_scan_results,
)


class GoldenSpiralTests(unittest.TestCase):
    def test_golden_spiral_g10_returns_two_n_plus_one_points(self):
        lon, lat = golden_spiral_g10(1)

        self.assertEqual(len(lon), 3)
        self.assertEqual(len(lat), 3)
        self.assertTrue(all(0.0 <= value < 360.0 for value in lon))
        self.assertTrue(all(-90.0 <= value <= 90.0 for value in lat))
        self.assertAlmostEqual(lon[1], 0.0)
        self.assertAlmostEqual(lat[1], 0.0)
        self.assertAlmostEqual(lat[0], math.degrees(math.asin(-2 / 3)))
        self.assertAlmostEqual(lat[2], math.degrees(math.asin(2 / 3)))

    def test_golden_spiral_g10_accepts_zero(self):
        lon, lat = golden_spiral_g10(0)

        self.assertEqual(list(lon), [0.0])
        self.assertEqual(list(lat), [0.0])

    def test_golden_spiral_g10_rejects_negative_values(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            golden_spiral_g10(-1)


class PoleScanSerializationTests(unittest.TestCase):
    def test_write_scan_results_and_best_result_store_expected_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            results = [
                PoleScanCandidateResult(
                    index=0,
                    initial_lambda=10.0,
                    initial_beta=-20.0,
                    status="failed",
                    error="convexinv failed",
                ),
                PoleScanCandidateResult(
                    index=1,
                    initial_lambda=30.0,
                    initial_beta=40.0,
                    status="success",
                    chi_square=1.5,
                    dev=0.2,
                    shadow_percent=0.3,
                    fitted_lambda=31.0,
                    fitted_beta=41.0,
                    fitted_period=5.0,
                    areas_file=str(output_dir / "areas.txt"),
                    lightcurve_output_file=str(output_dir / "lc.txt"),
                    param_file=str(output_dir / "params.txt"),
                    stdout_log_file=str(output_dir / "stdout.log"),
                ),
            ]

            csv_path = write_scan_results(results, output_dir / "scan.csv")
            json_path = write_best_result(results[1], output_dir / "best.json")

            with csv_path.open() as f:
                rows = list(csv.DictReader(f))
            best = json.loads(json_path.read_text())

        self.assertEqual(rows[0]["status"], "failed")
        self.assertEqual(rows[0]["error"], "convexinv failed")
        self.assertEqual(rows[1]["chi_square"], "1.5")
        self.assertEqual(rows[1]["shadow_percent"], "0.3")
        self.assertEqual(best["index"], 1)
        self.assertEqual(best["chi_square"], 1.5)


class PoleScanMapTests(unittest.TestCase):
    def test_build_pole_scan_map_figure_returns_separate_matplotlib_maps(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            results = [
                PoleScanCandidateResult(
                    index=0,
                    initial_lambda=10.0,
                    initial_beta=-20.0,
                    status="failed",
                    error="convexinv failed",
                ),
                PoleScanCandidateResult(
                    index=1,
                    initial_lambda=30.0,
                    initial_beta=40.0,
                    status="success",
                    chi_square=1.5,
                    shadow_percent=0.3,
                    fitted_period=5.0,
                ),
                PoleScanCandidateResult(
                    index=2,
                    initial_lambda=50.0,
                    initial_beta=-10.0,
                    status="success",
                    chi_square=2.5,
                    shadow_percent=0.6,
                    fitted_period=6.0,
                ),
            ]
            csv_path = write_scan_results(results, output_dir / "scan.csv")
            best_path = write_best_result(results[1], output_dir / "best.json")

            figs = build_pole_scan_map_figure(csv_path, best_path)

        self.assertEqual(len(figs), 3)
        for fig in figs:
            self.assertEqual(len(fig.axes), 2)
            self.assertIn(
                "Pole Grid Scan Map (initial pole coordinates)",
                fig._suptitle.get_text(),
            )
        self.assertEqual(
            [fig.axes[0].get_title() for fig in figs],
            [r"$\chi^2$", "Period [h]", "Dark facet area [%]"],
        )
        self.assertEqual(figs[0].axes[0].get_xlabel(), r"$\lambda$ [deg]")
        self.assertEqual(figs[0].axes[0].get_ylabel(), r"$\beta$ [deg]")
        self.assertTrue(any(r"$\chi^2$" in axis.get_ylabel() for axis in figs[0].axes))
        self.assertEqual(figs[0].axes[0].get_xlim(), (0.0, 360.0))
        self.assertEqual(figs[0].axes[0].get_ylim(), (-90.0, 90.0))

    def test_build_pole_scan_map_figure_can_plot_fitted_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            results = [
                PoleScanCandidateResult(
                    index=1,
                    initial_lambda=30.0,
                    initial_beta=40.0,
                    status="success",
                    chi_square=1.5,
                    shadow_percent=0.3,
                    fitted_lambda=130.0,
                    fitted_beta=-30.0,
                    fitted_period=5.0,
                ),
                PoleScanCandidateResult(
                    index=2,
                    initial_lambda=50.0,
                    initial_beta=-10.0,
                    status="success",
                    chi_square=2.5,
                    shadow_percent=0.6,
                    fitted_lambda=150.0,
                    fitted_beta=20.0,
                    fitted_period=6.0,
                ),
            ]
            csv_path = write_scan_results(results, output_dir / "scan.csv")
            best_path = write_best_result(results[0], output_dir / "best.json")

            figs = build_pole_scan_map_figure(
                csv_path, best_path, coordinate_mode="fitted"
            )

            point_offsets = figs[0].axes[0].collections[0].get_offsets().tolist()
            best_offsets = figs[0].axes[0].collections[1].get_offsets().tolist()

        self.assertIn("fitted pole coordinates", figs[0]._suptitle.get_text())
        self.assertEqual(point_offsets, [[130.0, -30.0], [150.0, 20.0]])
        self.assertEqual(best_offsets, [[130.0, -30.0]])

    def test_build_pole_scan_map_figure_drops_missing_fitted_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            results = [
                PoleScanCandidateResult(
                    index=1,
                    initial_lambda=30.0,
                    initial_beta=40.0,
                    status="success",
                    chi_square=1.5,
                    shadow_percent=0.3,
                    fitted_period=5.0,
                ),
                PoleScanCandidateResult(
                    index=2,
                    initial_lambda=50.0,
                    initial_beta=-10.0,
                    status="success",
                    chi_square=2.5,
                    shadow_percent=0.6,
                    fitted_lambda=150.0,
                    fitted_beta=20.0,
                    fitted_period=6.0,
                ),
            ]
            csv_path = write_scan_results(results, output_dir / "scan.csv")
            best_path = write_best_result(results[1], output_dir / "best.json")

            figs = build_pole_scan_map_figure(
                csv_path, best_path, coordinate_mode="fitted"
            )

            point_offsets = figs[0].axes[0].collections[0].get_offsets().tolist()

        self.assertEqual(point_offsets, [[150.0, 20.0]])

    def test_save_pole_scan_map_matplotlib_writes_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            results = [
                PoleScanCandidateResult(
                    index=1,
                    initial_lambda=30.0,
                    initial_beta=40.0,
                    status="success",
                    chi_square=1.5,
                    shadow_percent=0.3,
                    fitted_period=5.0,
                ),
                PoleScanCandidateResult(
                    index=2,
                    initial_lambda=50.0,
                    initial_beta=-10.0,
                    status="success",
                    chi_square=2.5,
                    shadow_percent=0.6,
                    fitted_period=6.0,
                ),
            ]
            csv_path = write_scan_results(results, output_dir / "scan.csv")
            best_path = write_best_result(results[0], output_dir / "best.json")
            png_paths = save_pole_scan_map_matplotlib(
                csv_path, best_path, output_dir / "pole_map.png"
            )

            png_bytes = [path.read_bytes() for path in png_paths]

        self.assertEqual(
            [path.name for path in png_paths],
            ["pole_map.png", "pole_map_period.png", "pole_map_shadow_percent.png"],
        )
        self.assertTrue(all(path.name.endswith(".png") for path in png_paths))
        self.assertTrue(all(data[:8] == b"\x89PNG\r\n\x1a\n" for data in png_bytes))


if __name__ == "__main__":
    unittest.main()
