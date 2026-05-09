import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .downloading import _download_lightcurve_url, _is_http_url
from .errors import AsteroidModelError
from .executables import run_convexinv, run_minkowski
from .lightcurves import (
    _write_lcs_dict_to_file,
    csv_to_lcs_format,
    dataframe_to_lcs_format,
    plot_lightcurves,
)
from .parameters import create_conjgradinv_param_file, create_convexinv_param_file
from .paths import CONVEXINV_EXEC, DAMIT_DIR, MINKOWSKI_EXEC, MODULE_DIR, PROJECT_ROOT
from .plotting import plot_model as _plot_model, plot_model_plotly as _plot_model_plotly
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
    "PROJECT_ROOT",
    "_download_lightcurve_url",
    "_is_http_url",
    "_parse_minkowski_output",
    "_triangulate_faces",
    "_write_lcs_dict_to_file",
    "create_conjgradinv_param_file",
    "create_convexinv_param_file",
    "csv_to_lcs_format",
    "dataframe_to_lcs_format",
    "load_model_obj",
    "plot_lightcurves",
    "plot_model",
    "plot_model_plotly",
    "plot_sky_projection",
    "plot_synthetic_lightcurve",
    "project_shape_to_sky",
    "run_convexinv",
    "run_minkowski",
    "run_pipeline",
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

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_lightcurves(self, source: Union[str, pd.DataFrame]):
        """
        Ingest lightcurve data from a pandas DataFrame, a CSV path, a native
        convexinv/DAMIT text file, or an http(s) URL to a native text file.
        """
        if isinstance(source, pd.DataFrame):
            self.lightcurves = source.copy()
            self.lightcurve_file = None
        elif isinstance(source, str) and source.lower().endswith(".csv"):
            self.lightcurves = pd.read_csv(source)
            self.lightcurve_file = None
            if "is_relative" not in self.lightcurves.columns:
                self.lightcurves["is_relative"] = 0
            if "curve_id" not in self.lightcurves.columns:
                self.lightcurves["curve_id"] = 1
        elif isinstance(source, str) and _is_http_url(source):
            self.lightcurves = None
            self.lightcurve_file = _download_lightcurve_url(source, self.output_dir)
        elif isinstance(source, str):
            if not Path(source).is_file():
                raise ValueError(f"Lightcurve file does not exist: {source}")
            self.lightcurves = None
            self.lightcurve_file = source
        else:
            raise ValueError(
                "Data source must be a pandas.DataFrame, a file path, or an http(s) lightcurve URL."
            )
        return self

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

    def run_inversion(self, verbose: bool = False):
        """
        Run convexinv and minkowski using the current modeler state.
        """
        if self.lightcurve_file is None and (
            self.lightcurves is None or self.lightcurves.empty
        ):
            raise AsteroidModelError(
                "No lightcurves loaded. Please call load_lightcurves() first."
            )

        base_name = self.asteroid_name.replace(" ", "_")

        try:
            tmp_param_file = str(self.output_dir / f"{base_name}_input_convexinv.txt")
            create_convexinv_param_file(self.inversion_options, tmp_param_file)

            tmp_conj_file = str(self.output_dir / "input_conjgradinv")
            create_conjgradinv_param_file(self.conjgradinv_options, tmp_conj_file)

            if self.lightcurve_file:
                tmp_lcs_file = self.lightcurve_file
                print(f"Using native convexinv lightcurve file: {tmp_lcs_file}")
            else:
                tmp_lcs_file = str(self.output_dir / f"{base_name}_lcs.txt")
                print("Converting DataFrame input to convexinv text format...")
                if self.lightcurves is None:
                    raise ValueError(
                        "No lightcurves provided. Supply a DataFrame or a lightcurve file."
                    )
                dataframe_to_lcs_format(self.lightcurves, tmp_lcs_file)

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

            print("Reconstructing 3D shape from generated features using minkowski...")
            self.vertices, self.faces = run_minkowski(
                actual_output_areas, pwd_dir=str(self.output_dir), verbose=verbose
            )

            print(
                f"Pipeline complete. All core metrics tracking {base_name} preserved in '{str(self.output_dir)}/'."
            )
            return self.vertices, self.faces

        except Exception as e:
            raise AsteroidModelError(f"Pipeline execution failed: {e}") from e

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
            str(self.output_dir / f"{base_name}_lightcurves.png") if save else None
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
        plot_lightcurves(
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
        plot_lcs_file = f"{base_name}_lightcurves.png"
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
        run_convexinv(
            param_file, actual_lightcurve_file, actual_output_areas, actual_output_lc
        )
        print("convexinv complete.")

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
