# Asteroid Modeling API Reference

This document provides detailed information about the functions available in the `asteroid_modeling` module. For an introduction and installation instructions, see the main [README.md](../README.md).

## Local Streamlit GUI

The repository includes a Streamlit app at `apps/streamlit_app.py`. It is a thin browser-based wrapper around `AsteroidModeler`, so the numerical behavior is the same as the Python API. Use the GUI for interactive local runs and the Python API for scripts, notebooks, and automated workflows. The GUI accepts DAMIT plaintext URLs or uploaded lightcurve files, exposes the DAMIT LSL scattering parameters, and saves sidebar values to `.pymit_last_params.json` after successful runs.

Pipeline execution runs in a background worker thread in the Streamlit app. The browser polls the thread-safe log state once per second, so stdout and stderr from the modeling pipeline appear while the job is still running.

The GUI can run `period_scan`, convex inversion, Minkowski reconstruction, pole-grid search, sky projection, and synthetic lightcurve generation. In the GUI, `p2`, `p3`, and `p4` have `Fit` checkboxes. These are off by default, which keeps the phase-function values fixed and is usually more stable for relative lightcurves.

## Module Layout

`pymit.asteroid_modeling` remains the public compatibility module. It owns `AsteroidModeler` and `run_pipeline`, and re-exports the helper functions that older code may import from that module.

Implementation details are split by responsibility:
-   `pymit.downloading`: URL detection and DAMIT lightcurve downloads.
-   `pymit.parameters`: `input_convexinv` and `input_conjgradinv` file generation.
-   `pymit.lightcurves`: DataFrame/CSV conversion and observed-vs-modeled lightcurve plotting.
-   `pymit.shape`: Minkowski output parsing, triangulation, and OBJ load/save helpers.
-   `pymit.plotting`: Matplotlib and Plotly 3D model plotting.
-   `pymit.projection`: sky-plane projection and fixed-geometry synthetic lightcurve helpers.
-   `pymit.period_scan`: `period_scan` output parsing, best-period selection, CSV export, and chi-square/RMS plot generation.
-   `pymit.pole_scan`: golden-spiral pole grids, scan result serialization, and pole-map plotting.
-   `pymit.executables`: subprocess wrappers for `convexinv` and `minkowski`.

## Core Classes

### `AsteroidModeler`
The main class to handle data state, configuration parameters, and the execution of the full end-to-end pipeline.

```python
class AsteroidModeler:
    def __init__(self, asteroid_name: str = "Asteroid", output_dir: str = "output"):
```
Initializes the modeler instance.
-   `asteroid_name` (str): The name used as a prefix for saved output files. Defaults to `"Asteroid"`.
-   `output_dir` (str): Directory where outputs are saved. Created if it doesn't exist. Defaults to `"output"`.

#### `load_lightcurves`
```python
    def load_lightcurves(self, source: Union[str, pd.DataFrame]):
```
Loads lightcurves into the modeler.
-   `source` (str or pd.DataFrame): Path to the observed light curves data file (`.csv` or native DAMIT/convexinv `.txt` / `.lc.txt` format), an `http(s)` URL to a native DAMIT plaintext lightcurve export, or a `pandas.DataFrame`.
-   Native text files are normalized before `convexinv` when their curve header counts do not match the actual rows. Repaired files are written as `*_convexinv.txt` in the modeler's output directory.

### `normalize_native_lcs_format`
```python
from pymit.asteroid_modeling import normalize_native_lcs_format

changed = normalize_native_lcs_format("data/153814_lcs4DAMIT_lite.txt", "output/153814_convexinv.txt")
```
Rewrites a native DAMIT/`convexinv` lightcurve file with internally consistent per-curve point counts and canonical A3301-style numeric formatting. It preserves the numeric values and returns `True` when the output differs from the input.

#### `load_parameters`
```python
    def load_parameters(
        self,
        inversion_json: Union[str, dict] = None,
        conjgradinv_json: Union[str, dict] = None
    ):
```
Loads inversion and shape construction options.
-   `inversion_json` (str or dict, optional): Configuration for the convex inversion step (`input_convexinv`). Can be a JSON file path, JSON string, or a dictionary.
-   `conjgradinv_json` (str or dict, optional): Configuration for the shape reconstruction step (`input_conjgradinv`).

Common `inversion_json` keys:
-   `initial_lambda`, `initial_beta`, `initial_period`: initial pole and period guesses.
-   `initial_lambda_fixed`, `initial_beta_fixed`, `initial_period_fixed`: `0` means fixed, `1` means free.
-   `phase_func_c`: DAMIT LSL `p1`, the Lambert coefficient `c`.
-   `phase_func_a`: DAMIT LSL `p2`, phase function amplitude `a`.
-   `phase_func_d`: DAMIT LSL `p3`, phase function width `d`.
-   `phase_func_k`: DAMIT LSL `p4`, phase function slope `k`.
-   `phase_func_*_fixed`: `0` means fixed, `1` means free. Keeping scattering parameters fixed is often more stable for relative lightcurves. Freeing `p2`/`phase_func_a`, `p3`/`phase_func_d`, or `p4`/`phase_func_k` requires enough phase-angle coverage; otherwise `convexinv` can fail with singular matrix errors.
-   `convexity_regularization`, `spherical_harmonics_degree`, `spherical_harmonics_order`, `number_of_rows`, `iteration_stop_condition`.

Common `conjgradinv_json` keys:
-   `convexity_weight`
-   `number_of_rows`
-   `number_of_iterations`

#### `run_inversion`
```python
    def run_inversion(
        self,
        verbose: bool = False,
        run_period_scan: bool = False,
        period_scan_options: dict | None = None,
        period_scan_param_file: str | None = None,
        period_scan_workers: int = 1,
    ) -> tuple[np.ndarray, list[list[int]]]:
```
Runs `convexinv` and `minkowski` using the loaded lightcurves and parameters. When `run_period_scan=True`, PyMit runs DAMIT `period_scan` first, writes raw/CSV/PNG period-scan outputs, stores the best result on `modeler.period_scan_result`, and uses that best period as `initial_period` for `convexinv`.
**Returns:**
-   `vertices` (np.ndarray): A numpy array of shape `(N, 3)` containing the X, Y, Z coordinates.
-   `faces` (list[list[int]]): A list where each element is a face, defined by a list of 1-based vertex indices.

#### `run_period_scan`
```python
    def run_period_scan(
        self,
        period_scan_options: dict | None = None,
        param_file: str | None = None,
        lightcurve_file: str | None = None,
        verbose: bool = False,
        workers: int = 1,
    ) -> PeriodScanResult:
```
Runs DAMIT `period_scan` without running reconstruction. If `param_file` is omitted, `period_scan_options` is written to `<asteroid>_input_period_scan.txt`. The raw output is saved as `<asteroid>_period_scan.txt`, parsed rows are saved as `<asteroid>_period_scan.csv`, and the chi-square/RMS plot is saved as `<asteroid>_period_scan.png`.

When `workers > 1`, the period range is split into contiguous subranges and merged back into one raw output file. The returned `PeriodScanResult` is the row with the smallest chi-square.

#### `run_pole_grid_scan`
```python
    def run_pole_grid_scan(
        self,
        n: int,
        workers: int = 1,
        reconstruct_best: bool = True,
        verbose: bool = False,
    ) -> tuple[np.ndarray, list[list[int]]]:
```
Runs `convexinv` over a golden-spiral grid of initial pole guesses before reconstruction. The grid uses `2N + 1` points from `golden_spiral_g10(n)`, where `initial_lambda = lon` and `initial_beta = lat`.

Each candidate keeps the current inversion settings except for `initial_lambda` and `initial_beta`. Results are written to `<asteroid>_pole_scan_results.csv`; the selected minimum-chi-square candidate is written to `<asteroid>_pole_scan_best.json`. A static Matplotlib pole-solution map is saved as `<asteroid>_pole_scan_map.png`. The dark/shadow facet percent is recorded for diagnostics but is not used for selection.

When `reconstruct_best=True`, the best candidate's areas, modeled lightcurve, and parameter file are copied to the standard output names and `minkowski` reconstructs the final model from that best run.

If successful candidates include fitted pole coordinates, PyMit also saves fitted-coordinate maps named `<asteroid>_pole_scan_map_fitted*.png`.

#### Shape and Plot Utils (Object Methods)
-   `plot_lightcurves_results(save=False, show=True, max_curves=3)`: Builds an interactive Plotly observed-vs-modeled brightness figure against solar phase angle when Sun/Earth vectors are available in the lightcurve input. If vectors are missing, it falls back to rotation phase when period and `t0` are available. With `save=True`, writes `<asteroid>_lightcurves.html`.
-   `plot_model(save=False, show=True)`: Renders the 3D shape using Matplotlib.
-   `plot_model_plotly(save=False, show=True)`: Renders the 3D shape as an interactive HTML file.
-   `plot_sky_projection(save=False, show=True, jd=..., phase_degrees=None)`: Saves or displays a 2D sky-plane projection of the current model. When `jd` is supplied, the modeler uses the fitted period and the exact first observation JD as `t0`.
-   `plot_synthetic_lightcurve(save=False, show=True, n_steps=72, jd=...)`: Computes and plots a fixed-geometry synthetic lightcurve and can save CSV/PNG outputs.
-   `export_obj()`: Exports the shape to `<output_dir>/<asteroid_name>.obj`.

---

## Utilities

### Period Scan Utilities

```python
from pymit.asteroid_modeling import (
    PeriodScanResult,
    find_best_period_scan_result,
    parse_period_scan_output,
    save_period_scan_plot,
    write_period_scan_csv,
)
```

-   `parse_period_scan_output(output_file)`: Parses DAMIT `period_scan` rows into `PeriodScanResult` objects.
-   `find_best_period_scan_result(results)`: Returns the row with the smallest chi-square.
-   `write_period_scan_csv(results, output_file)`: Writes parsed period-scan rows to CSV.
-   `save_period_scan_plot(results, output_file, best_result=None)`: Saves the chi-square/RMS plot and marks the selected period.

### `plot_model`
Visualizes the 3D shape model.

```python
def plot_model(
    vertices: np.ndarray, 
    faces: list[list[int]], 
    save_path: str = None, 
    show: bool = True
) -> None:
```

**Arguments:**
-   `vertices` (np.ndarray): The 3D coordinates of the vertices.
-   `faces` (list[list[int]]): The faces defining the polygons of the shape.
-   `save_path` (str, optional): The file path to save the generated plt plot.
-   `show` (bool): If `True`, displays the plot interactively using `plt.show()`.

### `plot_model_plotly`
Visualizes the 3D shape model interactively using plotly.

```python
def plot_model_plotly(
    vertices: np.ndarray, 
    faces: list[list[int]], 
    save_path: str = None, 
    show: bool = True
) -> None:
```

**Arguments:**
-   `vertices` (np.ndarray): The 3D coordinates of the vertices.
-   `faces` (list[list[int]]): The faces defining the polygons of the shape.
-   `save_path` (str, optional): The file path to save the generated HTML file.
-   `show` (bool): If `True`, opens the plot interactively in the browser.

### `load_model_obj`
Loads a previously generated 3D shape model directly from a `.obj` file. This lets you re-visualize and extract data without re-running the C and Fortran modeling codes.

```python
def load_model_obj(
    file_path: str
) -> tuple[np.ndarray, list[list[int]]]:
```

**Arguments:**
-   `file_path` (str): The absolute or relative path to the `.obj` file.

**Returns:**
-   `vertices` (np.ndarray): A numpy array of shape `(N, 3)` containing the X, Y, Z coordinates.
-   `faces` (list[list[int]]): A list where each element is a face defined by 1-based vertex indices.

### `save_model_obj`
Exports the 3D shape array into a `.obj` file format for broad compatibility.

```python
def save_model_obj(
    vertices: np.ndarray, 
    faces: list[list[int]], 
    save_path: str
) -> None:
```

**Arguments:**
-   `vertices` (np.ndarray): The 3D coordinates of the vertices.
-   `faces` (list[list[int]]): The faces defining the polygons of the shape.
-   `save_path` (str): The `.obj` file path to save.

### `project_shape_to_sky`
Rotates a shape model and projects it onto a 2D sky plane.

```python
def project_shape_to_sky(
    vertices: np.ndarray,
    faces: list[list[int]],
    *,
    jd: float = None,
    period_hours: float = None,
    zero_time: float = 0.0,
    initial_rotation_angle: float = 0.0,
    phase_degrees: float = None,
    view_vector=(0.0, 0.0, 1.0),
) -> SkyProjection:
```

Use `phase_degrees` to set the rotation phase directly. Otherwise pass `jd`, `period_hours`, and `zero_time`; `AsteroidModeler.plot_sky_projection()` fills `period_hours` from the fitted inversion result and `zero_time` from the exact first observation JD.

### `plot_sky_projection`
Plots the visible projected facets and returns the `SkyProjection` data object.

```python
def plot_sky_projection(
    vertices: np.ndarray,
    faces: list[list[int]],
    *,
    save_path: str = None,
    show: bool = True,
    **projection_kwargs,
) -> SkyProjection:
```

### `compute_synthetic_lightcurve`
Computes a simple fixed-geometry synthetic lightcurve from visible illuminated facets.

```python
def compute_synthetic_lightcurve(
    vertices: np.ndarray,
    faces: list[list[int]],
    *,
    n_steps: int = 72,
    jd: float = None,
    period_hours: float = None,
    zero_time: float = 0.0,
    initial_rotation_angle: float = 0.0,
    phase_degrees: float = None,
    sun_vector=(1.0, 0.0, 1.0),
    view_vector=(0.0, 0.0, 1.0),
    lambert_coefficient: float = 0.1,
) -> pd.DataFrame:
```

The returned DataFrame has `phase`, `phase_degrees`, and normalized `brightness` columns. This helper is useful for quick visualization; it is not a full replacement for the DAMIT scattering calculation.

### Projection CSV Helpers

```python
def save_sky_projection_csv(projection: SkyProjection, output_file: str) -> None:
def save_synthetic_lightcurve_csv(lightcurve: pd.DataFrame, output_file: str) -> None:
def first_observation_jd(lightcurve_source) -> float:
```

`first_observation_jd()` accepts a DataFrame or native lightcurve text file and returns the exact first observation JD without rounding.
