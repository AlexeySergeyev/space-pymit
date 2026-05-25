import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .downloading import _download_lightcurve_url, _is_http_url
from .errors import AsteroidModelError
from .executables import run_convexinv, run_minkowski, run_period_scan
from .lightcurves import (
    _write_lcs_dict_to_file,
    build_lightcurve_figure,
    csv_to_lcs_format,
    dataframe_to_lcs_format,
    normalize_native_lcs_format,
    plot_lightcurves,
    validate_lightcurve_dataframe_columns,
)
from .parameters import (
    create_conjgradinv_param_file,
    create_convexinv_param_file,
    create_period_scan_param_file,
)
from .paths import (
    CONVEXINV_EXEC,
    DAMIT_DIR,
    MINKOWSKI_EXEC,
    MODULE_DIR,
    PERIOD_SCAN_EXEC,
    PROJECT_ROOT,
)
from .period_scan import (
    PeriodScanResult,
    find_best_period_scan_result,
    parse_period_scan_output,
    save_period_scan_plot,
    write_period_scan_csv,
)
from .plotting import plot_model as _plot_model, plot_model_plotly as _plot_model_plotly
from .pole_scan import (
    PoleScanCandidateResult,
    golden_spiral_g10,
    save_pole_scan_map_matplotlib,
    write_best_result,
    write_scan_results,
)
from .projection import (
    compute_synthetic_lightcurve,
    first_observation_jd,
    plot_sky_projection as _plot_sky_projection,
    plot_synthetic_lightcurve as _plot_synthetic_lightcurve,
    project_shape_to_sky,
    save_sky_projection_csv,
    save_synthetic_lightcurve_csv,
)
from .shape import (
    _parse_minkowski_output,
    _triangulate_faces,
    load_model_obj,
    save_model_obj,
)


__all__ = [
    "AsteroidModelError",
    "AsteroidModeler",
    "CONVEXINV_EXEC",
    "DAMIT_DIR",
    "MINKOWSKI_EXEC",
    "MODULE_DIR",
    "PERIOD_SCAN_EXEC",
    "PROJECT_ROOT",
    "_download_lightcurve_url",
    "_is_http_url",
    "_parse_minkowski_output",
    "_triangulate_faces",
    "_write_lcs_dict_to_file",
    "create_conjgradinv_param_file",
    "create_convexinv_param_file",
    "create_period_scan_param_file",
    "build_lightcurve_figure",
    "csv_to_lcs_format",
    "dataframe_to_lcs_format",
    "normalize_native_lcs_format",
    "golden_spiral_g10",
    "load_model_obj",
    "plot_lightcurves",
    "plot_model",
    "plot_model_plotly",
    "plot_sky_projection",
    "plot_synthetic_lightcurve",
    "project_shape_to_sky",
    "run_convexinv",
    "run_minkowski",
    "run_period_scan",
    "run_pipeline",
    "PeriodScanResult",
    "find_best_period_scan_result",
    "parse_period_scan_output",
    "save_period_scan_plot",
    "write_period_scan_csv",
    "compute_synthetic_lightcurve",
    "first_observation_jd",
    "save_model_obj",
    "save_sky_projection_csv",
    "save_synthetic_lightcurve_csv",
]

# Public re-exports so `from pymit.asteroid_modeling import plot_model` etc. keep working
plot_model = _plot_model
plot_model_plotly = _plot_model_plotly
plot_sky_projection = _plot_sky_projection
plot_synthetic_lightcurve = _plot_synthetic_lightcurve


def _print_convexinv_fit_summary(fit_result: Optional[dict]) -> None:
    if not fit_result:
        print("convexinv fit summary: no parsed fit metrics were returned.", flush=True)
        return

    print("convexinv fit summary:", flush=True)
    if "chi_square" in fit_result:
        print(f"  chi-square: {fit_result['chi_square']}", flush=True)
    if "dev" in fit_result:
        print(f"  dev: {fit_result['dev']}", flush=True)
    if all(key in fit_result for key in ("lambda", "beta", "period")):
        print(
            "  lambda/beta/period: "
            f"{fit_result['lambda']} / {fit_result['beta']} / {fit_result['period']} h",
            flush=True,
        )
    if all(key in fit_result for key in ("phase_a", "phase_d", "phase_k")):
        print(
            "  phase a/d/k: "
            f"{fit_result['phase_a']} / {fit_result['phase_d']} / {fit_result['phase_k']}",
            flush=True,
        )
    if "lambert_c" in fit_result:
        print(f"  Lambert coefficient c: {fit_result['lambert_c']}", flush=True)
    if "shadow_percent" in fit_result:
        print(f"  dark facet area: {fit_result['shadow_percent']}%", flush=True)


def _has_successful_fitted_pole_coordinates(
    results: list[PoleScanCandidateResult],
) -> bool:
    return any(
        result.status == "success"
        and result.fitted_lambda is not None
        and result.fitted_beta is not None
        for result in results
    )


def _period_scan_bounds(param_file: Path) -> tuple[float, float]:
    first_line = param_file.read_text().splitlines()[0]
    parts = first_line.split()
    if len(parts) < 3:
        raise ValueError(f"Invalid period_scan parameter header in {param_file}.")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid period_scan range in {param_file}: {first_line}") from exc


def _split_period_range(start: float, end: float, workers: int) -> list[tuple[float, float]]:
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    if end <= start:
        raise ValueError("period_scan period_end must be greater than period_start.")
    if workers == 1:
        return [(start, end)]

    step = (end - start) / workers
    ranges = []
    for index in range(workers):
        range_start = start + index * step
        range_end = end if index == workers - 1 else start + (index + 1) * step
        ranges.append((range_start, range_end))
    return ranges


def _write_period_scan_param_range(
    source_file: Path,
    output_file: Path,
    period_start: float,
    period_end: float,
) -> None:
    lines = source_file.read_text().splitlines()
    if not lines:
        raise ValueError(f"Period scan parameter file is empty: {source_file}")
    first_parts = lines[0].split()
    if len(first_parts) < 3:
        raise ValueError(f"Invalid period_scan parameter header in {source_file}.")
    coefficient = first_parts[2]
    output_file.write_text(
        "\n".join(
            [
                f"{period_start:.12g} {period_end:.12g} {coefficient}\tperiod start - end - interval coeff.",
                *lines[1:],
            ]
        )
        + "\n"
    )


class AsteroidModeler:
    """
    Object-oriented wrapper for the PyMit asteroid modeling pipeline.

    This class owns workflow state: loaded lightcurves, inversion settings, generated
    shape vertices/faces, and output paths. Lower-level conversion, plotting, and
    executable helpers live in focused modules and are re-exported here.
    """

    def __init__(self, asteroid_name: str = "Asteroid", output_dir: str = "output"):
        self.asteroid_name = asteroid_name
        self.output_dir = Path(output_dir)
        self.lightcurves: Optional[pd.DataFrame] = None
        self.inversion_options: dict = {}
        self.conjgradinv_options: dict = {}
        self.lightcurve_file: Optional[str] = None
        self.vertices: Optional[np.ndarray] = None
        self.faces: Optional[list[list[int]]] = None
        self.fit_result: Optional[dict] = None
        self.period_scan_results: list[PeriodScanResult] = []
        self.period_scan_result: Optional[PeriodScanResult] = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_lightcurves(self, source: Union[str, pd.DataFrame]):
        """
        Ingest lightcurve data from a pandas DataFrame, a CSV path, a native
        convexinv/DAMIT text file, or an http(s) URL to a native text file.
        """
        if isinstance(source, pd.DataFrame):
            validate_lightcurve_dataframe_columns(source)
            self.lightcurves = source.copy()
            self.lightcurve_file = None
        elif isinstance(source, str) and source.lower().endswith(".csv"):
            self.lightcurves = pd.read_csv(source)
            validate_lightcurve_dataframe_columns(self.lightcurves, source_label="CSV")
            self.lightcurve_file = None
            if "is_relative" not in self.lightcurves.columns:
                self.lightcurves["is_relative"] = 0
            if "curve_id" not in self.lightcurves.columns:
                self.lightcurves["curve_id"] = 1
        elif isinstance(source, str) and _is_http_url(source):
            self.lightcurves = None
            self.lightcurve_file = self._prepare_native_lightcurve_file(
                _download_lightcurve_url(source, self.output_dir)
            )
        elif isinstance(source, str):
            if not Path(source).is_file():
                raise ValueError(f"Lightcurve file does not exist: {source}")
            self.lightcurves = None
            self.lightcurve_file = self._prepare_native_lightcurve_file(source)
        else:
            raise ValueError(
                "Data source must be a pandas.DataFrame, a file path, or an http(s) lightcurve URL."
            )
        return self

    def _prepare_native_lightcurve_file(self, source: str) -> str:
        source_path = Path(source)
        normalized_path = self.output_dir / f"{source_path.stem}_convexinv.txt"
        changed = normalize_native_lcs_format(source_path, normalized_path)
        if changed:
            return str(normalized_path)
        normalized_path.unlink(missing_ok=True)
        return str(source_path)

    def load_parameters(
        self,
        inversion_json: Optional[Union[str, dict]] = None,
        conjgradinv_json: Optional[Union[str, dict]] = None,
    ):
        """
        Load configuration parameters using JSON strings, file paths, or dictionaries.
        """

        def parse_json_input(data):
            if isinstance(data, dict):
                return data
            if isinstance(data, str):
                if Path(data).is_file():
                    with open(data, "r") as f:
                        return json.load(f)
                return json.loads(data)
            return {}

        if inversion_json is not None:
            self.inversion_options.update(parse_json_input(inversion_json))
        if conjgradinv_json is not None:
            self.conjgradinv_options.update(parse_json_input(conjgradinv_json))
        return self

    def run_inversion(
        self,
        verbose: bool = False,
        run_period_scan: bool = False,
        period_scan_options: Optional[dict] = None,
        period_scan_param_file: Optional[str] = None,
        period_scan_workers: int = 1,
    ):
        """
        Run convexinv and minkowski using the current modeler state.
        """
        base_name = self.asteroid_name.replace(" ", "_")

        try:
            self.run_convex_inversion(
                verbose=verbose,
                run_period_scan=run_period_scan,
                period_scan_options=period_scan_options,
                period_scan_param_file=period_scan_param_file,
                period_scan_workers=period_scan_workers,
            )
            self.run_minkowski_reconstruction(verbose=verbose)

            print(
                f"Pipeline complete. All core metrics tracking {base_name} preserved in '{str(self.output_dir)}/'."
            )
            return self.vertices, self.faces

        except Exception as e:
            raise AsteroidModelError(f"Pipeline execution failed: {e}") from e

    def run_convex_inversion(
        self,
        verbose: bool = False,
        run_period_scan: bool = False,
        period_scan_options: Optional[dict] = None,
        period_scan_param_file: Optional[str] = None,
        period_scan_workers: int = 1,
    ) -> dict:
        """
        Run convexinv and store the generated face-area file and fit metrics.
        """
        if self.lightcurve_file is None and (
            self.lightcurves is None or self.lightcurves.empty
        ):
            raise AsteroidModelError(
                "No lightcurves loaded. Please call load_lightcurves() first."
            )

        base_name = self.asteroid_name.replace(" ", "_")

        try:
            tmp_conj_file = str(self.output_dir / "input_conjgradinv")
            create_conjgradinv_param_file(self.conjgradinv_options, tmp_conj_file)

            tmp_lcs_file = self._prepare_lightcurve_for_convexinv(base_name)

            if run_period_scan:
                best_period = self.run_period_scan(
                    period_scan_options=period_scan_options,
                    param_file=period_scan_param_file,
                    lightcurve_file=tmp_lcs_file,
                    verbose=verbose,
                    workers=period_scan_workers,
                )
                self.inversion_options["initial_period"] = best_period.period_hours

            tmp_param_file = str(self.output_dir / f"{base_name}_input_convexinv.txt")
            create_convexinv_param_file(
                self._inversion_options_with_default_zero_time(), tmp_param_file
            )

            actual_output_areas = str(self.output_dir / f"{base_name}_areas.txt")
            actual_output_lc = str(self.output_dir / f"{base_name}_lc_output.csv")

            if verbose:
                print("Running convexinv...")
            else:
                print(
                    "Running convexinv... (this operates silently and may take a moment)"
                )
            self.fit_result = run_convexinv(
                tmp_param_file,
                tmp_lcs_file,
                actual_output_areas,
                actual_output_lc,
                verbose=verbose,
            )
            if self.period_scan_result is not None:
                self.fit_result["initial_period_from_period_scan"] = (
                    self.period_scan_result.period_hours
                )
            _print_convexinv_fit_summary(self.fit_result)
            return self.fit_result

        except Exception as e:
            raise AsteroidModelError(f"convexinv stage failed: {e}") from e

    def run_period_scan(
        self,
        period_scan_options: Optional[dict] = None,
        param_file: Optional[str] = None,
        lightcurve_file: Optional[str] = None,
        verbose: bool = False,
        workers: int = 1,
    ) -> PeriodScanResult:
        """
        Run DAMIT period_scan, save raw/CSV/plot outputs, and return the best period.
        """
        if self.lightcurve_file is None and (
            self.lightcurves is None or self.lightcurves.empty
        ):
            raise AsteroidModelError(
                "No lightcurves loaded. Please call load_lightcurves() first."
            )

        base_name = self.asteroid_name.replace(" ", "_")
        if lightcurve_file is None:
            lightcurve_file = self._prepare_lightcurve_for_convexinv(base_name)

        if workers < 1:
            raise ValueError("workers must be at least 1.")

        period_param_file = (
            Path(param_file)
            if param_file is not None
            else self.output_dir / f"{base_name}_input_period_scan.txt"
        )
        if param_file is None:
            create_period_scan_param_file(period_scan_options or {}, str(period_param_file))

        raw_output_file = self.output_dir / f"{base_name}_period_scan.txt"
        stdout_log_file = self.output_dir / f"{base_name}_period_scan_stdout.log"
        print("Running period_scan to estimate the initial period...")
        if workers == 1:
            run_period_scan(
                str(period_param_file),
                lightcurve_file,
                str(raw_output_file),
                verbose=verbose,
                stdout_log_file=str(stdout_log_file),
            )
        else:
            self._run_period_scan_workers(
                period_param_file,
                lightcurve_file,
                raw_output_file,
                stdout_log_file,
                workers=workers,
                verbose=verbose,
            )

        results = parse_period_scan_output(raw_output_file)
        best = find_best_period_scan_result(results)
        csv_file = self.output_dir / f"{base_name}_period_scan.csv"
        plot_file = self.output_dir / f"{base_name}_period_scan.png"
        write_period_scan_csv(results, csv_file)
        save_period_scan_plot(results, plot_file, best_result=best)

        self.period_scan_results = results
        self.period_scan_result = best
        self.inversion_options["initial_period"] = best.period_hours
        print(
            "period_scan best period: "
            f"{best.period_hours} h (chi-square {best.chi_square})."
        )
        return best

    def _run_period_scan_workers(
        self,
        period_param_file: Path,
        lightcurve_file: str,
        raw_output_file: Path,
        stdout_log_file: Path,
        workers: int,
        verbose: bool = False,
    ) -> None:
        start, end = _period_scan_bounds(period_param_file)
        ranges = _split_period_range(start, end, workers)
        worker_dir = self.output_dir / "period_scan"
        worker_dir.mkdir(parents=True, exist_ok=True)

        def run_worker(index: int, period_start: float, period_end: float) -> tuple[int, Path, Path]:
            param_file = worker_dir / f"input_period_scan_{index:04d}.txt"
            output_file = worker_dir / f"period_scan_{index:04d}.txt"
            log_file = worker_dir / f"period_scan_stdout_{index:04d}.log"
            _write_period_scan_param_range(period_param_file, param_file, period_start, period_end)
            run_period_scan(
                str(param_file),
                lightcurve_file,
                str(output_file),
                verbose=verbose,
                stdout_log_file=str(log_file),
            )
            return index, output_file, log_file

        completed = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(run_worker, index, period_start, period_end): index
                for index, (period_start, period_end) in enumerate(ranges)
            }
            for future in as_completed(future_map):
                completed.append(future.result())
        completed.sort(key=lambda item: item[0])

        raw_output_file.write_text(
            "".join(output_file.read_text() for _index, output_file, _log_file in completed)
        )
        stdout_log_file.write_text(
            "".join(log_file.read_text() for _index, _output_file, log_file in completed)
        )

    def run_minkowski_reconstruction(self, verbose: bool = False):
        """
        Run minkowski reconstruction from the most recent convexinv areas file.
        """
        base_name = self.asteroid_name.replace(" ", "_")
        actual_output_areas = str(self.output_dir / f"{base_name}_areas.txt")

        try:
            print("Reconstructing 3D shape from generated features using minkowski...")
            self.vertices, self.faces = run_minkowski(
                actual_output_areas, pwd_dir=str(self.output_dir), verbose=verbose
            )
            return self.vertices, self.faces

        except Exception as e:
            raise AsteroidModelError(f"minkowski stage failed: {e}") from e

    def run_pole_grid_scan(
        self,
        n: int,
        workers: int = 1,
        reconstruct_best: bool = True,
        verbose: bool = False,
    ):
        """Run convexinv over a golden-spiral pole grid and reconstruct the best fit."""
        if workers < 1:
            raise ValueError("workers must be at least 1.")
        if self.lightcurve_file is None and (
            self.lightcurves is None or self.lightcurves.empty
        ):
            raise AsteroidModelError(
                "No lightcurves loaded. Please call load_lightcurves() first."
            )

        base_name = self.asteroid_name.replace(" ", "_")
        lon_values, lat_values = golden_spiral_g10(n)
        tmp_lcs_file = self._prepare_lightcurve_for_convexinv(base_name)
        create_conjgradinv_param_file(
            self.conjgradinv_options, str(self.output_dir / "input_conjgradinv")
        )

        scan_dir = self.output_dir / "pole_scan"
        scan_dir.mkdir(parents=True, exist_ok=True)

        def run_candidate(index: int, initial_lambda: float, initial_beta: float):
            run_dir = scan_dir / f"run_{index:04d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            param_file = run_dir / "input_convexinv.txt"
            areas_file = run_dir / "areas.txt"
            output_lc_file = run_dir / "lc_output.csv"
            stdout_log_file = run_dir / "convexinv_stdout.log"

            options = {
                **self._inversion_options_with_default_zero_time(),
                "initial_lambda": float(initial_lambda),
                "initial_beta": float(initial_beta),
            }
            create_convexinv_param_file(options, str(param_file))

            try:
                fit = run_convexinv(
                    str(param_file),
                    tmp_lcs_file,
                    str(areas_file),
                    str(output_lc_file),
                    verbose=verbose,
                    stdout_log_file=str(stdout_log_file),
                )
            except Exception as exc:
                return PoleScanCandidateResult(
                    index=index,
                    initial_lambda=float(initial_lambda),
                    initial_beta=float(initial_beta),
                    status="failed",
                    param_file=str(param_file),
                    stdout_log_file=str(stdout_log_file),
                    error=str(exc),
                )

            return PoleScanCandidateResult(
                index=index,
                initial_lambda=float(initial_lambda),
                initial_beta=float(initial_beta),
                status="success",
                chi_square=fit.get("chi_square"),
                dev=fit.get("dev"),
                shadow_percent=fit.get("shadow_percent"),
                fitted_lambda=fit.get("lambda"),
                fitted_beta=fit.get("beta"),
                fitted_period=fit.get("period"),
                areas_file=str(areas_file),
                lightcurve_output_file=str(output_lc_file),
                param_file=str(param_file),
                stdout_log_file=str(stdout_log_file),
                fit_result=fit,
            )

        jobs = [
            (index, float(initial_lambda), float(initial_beta))
            for index, (initial_lambda, initial_beta) in enumerate(
                zip(lon_values, lat_values)
            )
        ]

        if workers == 1:
            results = [run_candidate(*job) for job in jobs]
        else:
            results = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(run_candidate, *job): job[0] for job in jobs
                }
                for future in as_completed(future_map):
                    results.append(future.result())
        results.sort(key=lambda result: result.index)

        results_csv = self.output_dir / f"{base_name}_pole_scan_results.csv"
        write_scan_results(results, results_csv)

        successful_results = [
            result
            for result in results
            if result.status == "success" and result.chi_square is not None
        ]
        if not successful_results:
            raise AsteroidModelError(
                f"Pole grid scan did not produce any successful chi-square results. See {results_csv}."
            )

        if not reconstruct_best:
            best = min(successful_results, key=lambda result: result.chi_square)
            best_json = write_best_result(
                best, self.output_dir / f"{base_name}_pole_scan_best.json"
            )
            save_pole_scan_map_matplotlib(
                results_csv,
                best_json,
                self.output_dir / f"{base_name}_pole_scan_map.png",
            )
            if _has_successful_fitted_pole_coordinates(results):
                save_pole_scan_map_matplotlib(
                    results_csv,
                    best_json,
                    self.output_dir / f"{base_name}_pole_scan_map_fitted.png",
                    coordinate_mode="fitted",
                )
            self.fit_result = dict(best.fit_result)
            self.fit_result["pole_scan_index"] = best.index
            self.fit_result["initial_lambda"] = best.initial_lambda
            self.fit_result["initial_beta"] = best.initial_beta
            return results

        standard_areas_file = self.output_dir / f"{base_name}_areas.txt"
        standard_lc_file = self.output_dir / f"{base_name}_lc_output.csv"
        standard_param_file = self.output_dir / f"{base_name}_input_convexinv.txt"

        best = None
        reconstruction_errors = []
        for candidate in sorted(successful_results, key=lambda result: result.chi_square):
            if (
                candidate.areas_file is None
                or candidate.lightcurve_output_file is None
                or candidate.param_file is None
            ):
                error = "Pole scan candidate is missing output artifacts."
                results[candidate.index] = replace(
                    candidate, status="reconstruction_failed", error=error
                )
                reconstruction_errors.append(f"candidate {candidate.index}: {error}")
                continue

            print(
                "Reconstructing 3D shape from pole-grid candidate "
                f"{candidate.index} (chi-square {candidate.chi_square})..."
            )
            try:
                vertices, faces = run_minkowski(
                    candidate.areas_file, pwd_dir=str(self.output_dir), verbose=verbose
                )
            except Exception as exc:
                error = f"minkowski reconstruction failed: {exc}"
                results[candidate.index] = replace(
                    candidate, status="reconstruction_failed", error=error
                )
                reconstruction_errors.append(f"candidate {candidate.index}: {exc}")
                continue

            best = candidate
            self.vertices, self.faces = vertices, faces
            break

        write_scan_results(results, results_csv)
        if best is None:
            raise AsteroidModelError(
                "Pole grid scan did not produce any reconstructable shape. "
                f"See {results_csv}. Reconstruction errors: "
                + "; ".join(reconstruction_errors)
            )

        best_json = write_best_result(
            best, self.output_dir / f"{base_name}_pole_scan_best.json"
        )
        save_pole_scan_map_matplotlib(
            results_csv,
            best_json,
            self.output_dir / f"{base_name}_pole_scan_map.png",
        )
        if _has_successful_fitted_pole_coordinates(results):
            save_pole_scan_map_matplotlib(
                results_csv,
                best_json,
                self.output_dir / f"{base_name}_pole_scan_map_fitted.png",
                coordinate_mode="fitted",
            )
        self.fit_result = dict(best.fit_result)
        self.fit_result["pole_scan_index"] = best.index
        self.fit_result["initial_lambda"] = best.initial_lambda
        self.fit_result["initial_beta"] = best.initial_beta

        shutil.copyfile(best.areas_file, standard_areas_file)
        shutil.copyfile(best.lightcurve_output_file, standard_lc_file)
        shutil.copyfile(best.param_file, standard_param_file)
        return self.vertices, self.faces

    def _prepare_lightcurve_for_convexinv(self, base_name: str) -> str:
        if self.lightcurve_file:
            print(f"Using native convexinv lightcurve file: {self.lightcurve_file}")
            return self.lightcurve_file

        tmp_lcs_file = str(self.output_dir / f"{base_name}_lcs.txt")
        print("Converting DataFrame input to convexinv text format...")
        if self.lightcurves is None:
            raise ValueError(
                "No lightcurves provided. Supply a DataFrame or a lightcurve file."
            )
        dataframe_to_lcs_format(self.lightcurves, tmp_lcs_file)
        return tmp_lcs_file

    def _inversion_options_with_default_zero_time(self) -> dict:
        options = dict(self.inversion_options)
        if "zero_time" in options:
            return options

        if self.lightcurve_file is not None:
            lightcurve_input = self.lightcurve_file
        elif self.lightcurves is not None:
            lightcurve_input = self.lightcurves
        else:
            raise AsteroidModelError(
                "No lightcurves loaded. Please call load_lightcurves() first."
            )
        options["zero_time"] = round(first_observation_jd(lightcurve_input))
        return options

    def plot_lightcurves_results(
        self, save: bool = False, show: bool = True, max_curves: int = 3
    ):
        """
        Plot modeled output against the loaded lightcurve data.
        """
        if self.lightcurve_file is None and (
            self.lightcurves is None or self.lightcurves.empty
        ):
            raise AsteroidModelError(
                "No lightcurves loaded. Please call load_lightcurves() first."
            )

        base_name = self.asteroid_name.replace(" ", "_")
        actual_output_lc = str(self.output_dir / f"{base_name}_lc_output.csv")
        save_path = (
            str(self.output_dir / f"{base_name}_lightcurves.html") if save else None
        )
        if self.lightcurve_file is not None:
            lightcurve_input = self.lightcurve_file
        else:
            lightcurve_input = self.lightcurves
            if lightcurve_input is None:
                raise AsteroidModelError(
                    "No lightcurves loaded. Please call load_lightcurves() first."
                )
        period_hours = (
            self.fit_result.get("period")
            if self.fit_result and "period" in self.fit_result
            else None
        )
        zero_time = first_observation_jd(lightcurve_input)

        print("Plotting lightcurves...")
        return plot_lightcurves(
            lightcurve_input,
            actual_output_lc,
            save_path=save_path,
            show=show,
            max_curves=max_curves,
            period_hours=period_hours,
            zero_time=zero_time,
        )

    def plot_model(self, save: bool = False, show: bool = True):
        """Visualizes the internal 3D model footprint using Matplotlib."""
        if self.vertices is None or self.faces is None:
            raise AsteroidModelError(
                "3D Model not yet generated. Run run_inversion() first."
            )

        base_name = self.asteroid_name.replace(" ", "_")
        save_path = str(self.output_dir / f"{base_name}_model.png") if save else None

        _plot_model(self.vertices, self.faces, save_path=save_path, show=show)

    def plot_model_plotly(self, save: bool = False, show: bool = True):
        """Visualizes the internal 3D model footprint using interactive Plotly HTML."""
        if self.vertices is None or self.faces is None:
            raise AsteroidModelError(
                "3D Model not yet generated. Run run_inversion() first."
            )

        base_name = self.asteroid_name.replace(" ", "_")
        save_path = str(self.output_dir / f"{base_name}_model.html") if save else None

        _plot_model_plotly(self.vertices, self.faces, save_path=save_path, show=show)

    def export_obj(self):
        """Exports the internal 3D representation to a standard Wavefront (.obj) file."""
        if self.vertices is None or self.faces is None:
            raise AsteroidModelError(
                "3D Model not yet generated. Run run_inversion() first."
            )

        base_name = self.asteroid_name.replace(" ", "_")
        save_path = str(self.output_dir / f"{base_name}.obj")

        save_model_obj(self.vertices, self.faces, save_path)
        print(f"Model successfully exported to {save_path}")

    def plot_sky_projection(
        self,
        save: bool = False,
        show: bool = True,
        **projection_kwargs,
    ):
        """Project the generated model onto the sky plane and optionally save PNG/CSV outputs."""
        if self.vertices is None or self.faces is None:
            raise AsteroidModelError(
                "3D Model not yet generated. Run run_inversion() first."
            )

        self._fill_projection_defaults(projection_kwargs)
        base_name = self.asteroid_name.replace(" ", "_")
        save_path = (
            str(self.output_dir / f"{base_name}_sky_projection.png") if save else None
        )
        projection = _plot_sky_projection(
            self.vertices,
            self.faces,
            save_path=save_path,
            show=show,
            **projection_kwargs,
        )
        if save:
            csv_path = str(self.output_dir / f"{base_name}_sky_projection.csv")
            save_sky_projection_csv(projection, csv_path)
        return projection

    def plot_synthetic_lightcurve(
        self,
        save: bool = False,
        show: bool = True,
        n_steps: int = 72,
        **lightcurve_kwargs,
    ):
        """Compute and plot a fixed-geometry synthetic light curve from the model."""
        if self.vertices is None or self.faces is None:
            raise AsteroidModelError(
                "3D Model not yet generated. Run run_inversion() first."
            )

        self._fill_projection_defaults(lightcurve_kwargs)
        if "lambert_coefficient" not in lightcurve_kwargs:
            lightcurve_kwargs["lambert_coefficient"] = self.inversion_options.get(
                "phase_func_c", 0.1
            )

        base_name = self.asteroid_name.replace(" ", "_")
        lightcurve = compute_synthetic_lightcurve(
            self.vertices, self.faces, n_steps=n_steps, **lightcurve_kwargs
        )
        save_path = (
            str(self.output_dir / f"{base_name}_synthetic_lightcurve.png")
            if save
            else None
        )
        _plot_synthetic_lightcurve(lightcurve, save_path=save_path, show=show)
        if save:
            csv_path = str(self.output_dir / f"{base_name}_synthetic_lightcurve.csv")
            save_synthetic_lightcurve_csv(lightcurve, csv_path)
        return lightcurve

    def _fill_projection_defaults(self, projection_kwargs: dict) -> None:
        if "phase_degrees" in projection_kwargs:
            return
        if "jd" not in projection_kwargs or projection_kwargs["jd"] is None:
            raise AsteroidModelError("Julian Date is required for projection.")
        if "period_hours" not in projection_kwargs:
            if not self.fit_result or "period" not in self.fit_result:
                raise AsteroidModelError(
                    "Computed period is unavailable. Run run_inversion() before projection."
                )
            projection_kwargs["period_hours"] = self.fit_result["period"]
        if "zero_time" not in projection_kwargs:
            if self.lightcurve_file is not None:
                projection_kwargs["zero_time"] = first_observation_jd(self.lightcurve_file)
            elif self.lightcurves is not None:
                projection_kwargs["zero_time"] = first_observation_jd(self.lightcurves)
            else:
                raise AsteroidModelError(
                    "Observation t0 is unavailable. Load lightcurves before projection."
                )
        if "initial_rotation_angle" not in projection_kwargs:
            projection_kwargs["initial_rotation_angle"] = self.inversion_options.get(
                "initial_rotation_angle", 0.0
            )


def run_pipeline(
    lightcurve: Union[str, pd.DataFrame],
    output_areas_file: Optional[str] = None,
    output_lc_file: Optional[str] = None,
    plot_file: Optional[Union[str, bool]] = None,
    obj_file: Optional[Union[str, bool]] = None,
    plotly_file: Optional[Union[str, bool]] = None,
    plot_lcs_file: Optional[Union[str, bool]] = None,
    output_dir: Optional[str] = None,
    asteroid_name: Optional[str] = None,
    param_file: Optional[str] = None,
    inversion_options: Optional[dict] = None,
    conjgradinv_options: Optional[dict] = None,
) -> tuple[np.ndarray, list[list[int]]]:
    """
    Run the full pipeline: compute areas from light curves, reconstruct 3D shape, and visualize.
    """
    base_name = asteroid_name if asteroid_name else "asteroid"

    if output_areas_file is None:
        output_areas_file = f"{base_name}_areas.txt"
    if output_lc_file is None:
        output_lc_file = f"{base_name}_lcs.txt"

    if plot_file is True:
        plot_file = f"{base_name}_model.png"
    elif plot_file is False:
        plot_file = None

    if obj_file is True:
        obj_file = f"{base_name}_model.obj"
    elif obj_file is False:
        obj_file = None

    if plotly_file is True:
        plotly_file = f"{base_name}_model.html"
    elif plotly_file is False:
        plotly_file = None

    if plot_lcs_file is True:
        plot_lcs_file = f"{base_name}_lightcurves.html"
    elif plot_lcs_file is False:
        plot_lcs_file = None

    out_dir = Path(output_dir) if output_dir else Path(".")
    if output_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    actual_output_areas = str(out_dir / output_areas_file)
    actual_output_lc = str(out_dir / output_lc_file)
    actual_lightcurve_file = lightcurve if isinstance(lightcurve, str) else None

    tmp_conj_file = None

    if inversion_options is not None:
        print("Generating dynamic input_convexinv settings...")
        param_file = str(out_dir / f"{base_name}_input_convexinv.txt")
        create_convexinv_param_file(inversion_options, param_file)
    elif not param_file:
        raise ValueError(
            "Must provide either a 'param_file' path or 'inversion_options' dict."
        )

    if conjgradinv_options is not None:
        print("Generating dynamic input_conjgradinv settings...")
        tmp_conj_file = str(out_dir / "input_conjgradinv")
        create_conjgradinv_param_file(conjgradinv_options, tmp_conj_file)

    if isinstance(lightcurve, pd.DataFrame):
        print("Converting DataFrame input to convexinv text format...")
        actual_lightcurve_file = str(out_dir / f"{base_name}_lcs_input.txt")
        dataframe_to_lcs_format(lightcurve, actual_lightcurve_file)
    elif isinstance(lightcurve, str) and _is_http_url(lightcurve):
        print(f"Downloading DAMIT lightcurve {lightcurve}...")
        actual_lightcurve_file = _download_lightcurve_url(lightcurve, out_dir)
    elif isinstance(lightcurve, str) and lightcurve.lower().endswith(".csv"):
        print(f"Converting CSV input {lightcurve} to convexinv text format...")
        filename = Path(lightcurve).with_suffix(".txt").name
        actual_lightcurve_file = str(out_dir / filename)
        csv_to_lcs_format(lightcurve, actual_lightcurve_file)
    elif isinstance(lightcurve, str):
        actual_lightcurve_file = lightcurve
    else:
        raise ValueError(
            "lightcurve must be a string file path, a CSV file path, or a pandas DataFrame."
        )

    if actual_lightcurve_file is None:
        raise ValueError("No lightcurve input file was prepared.")

    try:
        print("Running convexinv... (this might take a few moments)")
        fit_result = run_convexinv(
            param_file, actual_lightcurve_file, actual_output_areas, actual_output_lc
        )
        print("convexinv complete.")
        _print_convexinv_fit_summary(fit_result)

        print("Running minkowski 3D reconstruction...")
        vertices, faces = run_minkowski(
            actual_output_areas, pwd_dir=out_dir if tmp_conj_file else None
        )
        print(f"Reconstruction complete: {len(vertices)} vertices, {len(faces)} faces.")

        if obj_file:
            actual_obj = str(out_dir / obj_file)
            print(f"Saving 3D model to {actual_obj}...")
            save_model_obj(vertices, faces, save_path=actual_obj)

        if plot_file:
            actual_plot = str(out_dir / plot_file)
            print(f"Plotting matplotlib model to {actual_plot}...")
            plot_model(vertices, faces, save_path=actual_plot, show=False)

        if plotly_file:
            actual_plotly = str(out_dir / plotly_file)
            print(f"Generating interactive plotly model to {actual_plotly}...")
            plot_model_plotly(vertices, faces, save_path=actual_plotly, show=False)

        if plot_lcs_file:
            actual_plot_lcs = str(out_dir / plot_lcs_file)
            print(f"Plotting lightcurves to {actual_plot_lcs}...")
            plot_lightcurves(
                actual_lightcurve_file,
                actual_output_lc,
                save_path=actual_plot_lcs,
                show=False,
            )

        print(f"All intermediate files preserved in '{out_dir}/'.")
        return vertices, faces

    except Exception as e:
        raise AsteroidModelError(f"Pipeline execution failed: {e}") from e


if __name__ == "__main__":
    param_txt = DAMIT_DIR / "input_convexinv"
    lcs_txt = DAMIT_DIR / "test_lcs_abs"

    if param_txt.exists() and lcs_txt.exists():
        print("Testing file-based pipeline execution:")
        run_pipeline(
            lightcurve=str(lcs_txt),
            plot_file=True,
            obj_file=True,
            plotly_file=True,
            plot_lcs_file=True,
            output_dir="pipeline_output",
            asteroid_name="test_asteroid",
            param_file=str(param_txt),
            conjgradinv_options={"number_of_iterations": 150},
        )

        test_csv = PROJECT_ROOT / "pipeline_output" / "test.csv"
        if test_csv.exists():
            print("\nTesting DataFrame-based pipeline execution:")
            df = pd.read_csv(str(test_csv))
            run_pipeline(
                lightcurve=df,
                plot_file=False,
                obj_file=False,
                plotly_file=False,
                plot_lcs_file=False,
                output_dir="pipeline_output",
                asteroid_name="test_asteroid_df",
                param_file=str(param_txt),
            )
    else:
        print("Example data files not found. Please provide them to test the pipeline.")
