# Asteroid Modeling API Reference

This document provides detailed information about the functions available in the `asteroid_modeling` module. For an introduction and installation instructions, see the main [README.md](../README.md).

## Local GUI

The repository includes a Streamlit app at `apps/streamlit_app.py`. It is a thin browser-based wrapper around `AsteroidModeler`, so the numerical behavior is the same as the Python API. Use the GUI for interactive local runs and the Python API for scripts, notebooks, and automated workflows. The GUI accepts DAMIT plaintext URLs or uploaded lightcurve files, exposes the DAMIT LSL scattering parameters, and saves sidebar values to `.pymit_last_params.json` after successful runs.

## Module Layout

`pymit.asteroid_modeling` remains the public compatibility module. It owns `AsteroidModeler` and `run_pipeline`, and re-exports the helper functions that older code may import from that module.

Implementation details are split by responsibility:
-   `pymit.downloading`: URL detection and DAMIT lightcurve downloads.
-   `pymit.parameters`: `input_convexinv` and `input_conjgradinv` file generation.
-   `pymit.lightcurves`: DataFrame/CSV conversion and observed-vs-modeled lightcurve plotting.
-   `pymit.shape`: Minkowski output parsing, triangulation, and OBJ load/save helpers.
-   `pymit.plotting`: Matplotlib and Plotly 3D model plotting.
-   `pymit.projection`: sky-plane projection and fixed-geometry synthetic lightcurve helpers.
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
-   `phase_func_*_fixed`: `0` means fixed, `1` means free. Keeping scattering parameters fixed is often more stable for relative lightcurves.
-   `convexity_regularization`, `spherical_harmonics_degree`, `spherical_harmonics_order`, `number_of_rows`, `iteration_stop_condition`.

Common `conjgradinv_json` keys:
-   `convexity_weight`
-   `number_of_rows`
-   `number_of_iterations`

#### `run_inversion`
```python
    def run_inversion(self) -> tuple[np.ndarray, list[list[int]]]:
```
Executes the shape reconstruction using the loaded lightcurves and parameters.
**Returns:**
-   `vertices` (np.ndarray): A numpy array of shape `(N, 3)` containing the X, Y, Z coordinates.
-   `faces` (list[list[int]]): A list where each element is a face, defined by a list of 1-based vertex indices.

#### Shape and Plot Utils (Object Methods)
-   `plot_lightcurves_results(save=False, show=True, max_curves=3)`: Plots observed vs modeled brightness against solar phase angle when Sun/Earth vectors are available in the lightcurve input. If vectors are missing, it falls back to rotation phase when period and `t0` are available.
-   `plot_model(save=False, show=True)`: Renders the 3D shape using Matplotlib.
-   `plot_model_plotly(save=False, show=True)`: Renders the 3D shape as an interactive HTML file.
-   `plot_sky_projection(save=False, show=True, jd=..., phase_degrees=None)`: Saves or displays a 2D sky-plane projection of the current model. When `jd` is supplied, the modeler uses the fitted period and the exact first observation JD as `t0`.
-   `plot_synthetic_lightcurve(save=False, show=True, n_steps=72, jd=...)`: Computes and plots a fixed-geometry synthetic lightcurve and can save CSV/PNG outputs.
-   `export_obj()`: Exports the shape to `<output_dir>/<asteroid_name>.obj`.

---

## Utilities

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
