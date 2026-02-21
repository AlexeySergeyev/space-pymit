# Asteroid Modeling API Reference

This document provides detailed information about the functions available in the `asteroid_modeling` module. For an introduction and installation instructions, see the main [README.md](../README.md).

## Core Functions

### `run_pipeline`
The core function to execute the full end-to-end pipeline.

```python
def run_pipeline(
    lightcurve: Union[str, pd.DataFrame], 
    output_areas_file: str = None, 
    output_lc_file: str = None, 
    plot_file: Union[str, bool] = None,
    obj_file: Union[str, bool] = None,
    plotly_file: Union[str, bool] = None,
    plot_lcs_file: Union[str, bool] = None,
    output_dir: str = None,
    asteroid_name: str = None,
    param_file: str = None,
    inversion_options: dict = None,
    conjgradinv_options: dict = None
) -> tuple[np.ndarray, list[list[int]]]:
```

**Arguments:**
-   `lightcurve` (str or pd.DataFrame): Path to the observed light curves data file (txt or csv) OR a `pandas.DataFrame`. If a `.csv` file or `DataFrame` is provided, it is automatically converted into the target text format before execution. The expected columns are `jd`, `brightness`, `sun_x`, `sun_y`, `sun_z`, `earth_x`, `earth_y`, `earth_z`, with optional `curve_id` and `is_relative` logic mapping lines into distinct light curves.
-   `output_areas_file` (str, optional): Path where the intermediate output containing areas and normals will be saved. Defaults to `{asteroid_name}_areas.txt` if `asteroid_name` is provided, otherwise `output_areas.txt`.
-   `output_lc_file` (str, optional): Path where the modeled (fitted) light curves will be output. Defaults to `{asteroid_name}_lcs.txt` if string formatting is leveraged, otherwise `output_lcs.txt`.
-   `plot_file` (str or bool, optional): If provided as a string, saves a static matplotlib visualization to this given filename. If `True`, generates a default `{asteroid_name}_model.png`.
-   `obj_file` (str or bool, optional): If provided as a string, saves the exact 3D polygon shape to a standard Wavefront `.obj`. If `True`, generates a default `.obj` file.
-   `plotly_file` (str or bool, optional): If provided as a string, saves an interactive HTML 3D visualization. If `True`, generates a default `.html` template based on the asteroid.
-   `plot_lcs_file` (str or bool, optional): If provided as a string, saves a scatter plot of the observed vs modeled light curves. If `True`, generates a default `{asteroid_name}_lightcurves.png`.
-   `output_dir` (str, optional): A directory where all the specified outputs (`output_areas_file`, `output_lc_file`, plots, and models) will be saved. If it doesn't exist, it will be created. If not provided, it saves to the current working directory.
-   `asteroid_name` (str, optional): Used to dynamically fill output filenames if no strict path is provided. Defaults to "asteroid".
-   `param_file` (str, optional): Path to the `input_convexinv` configuration file. This file contains parameters like initial period, regularizations, and iteration limits.
-   `inversion_options` (dict, optional): Overrides individual numerical tuning flags to construct the `input_convexinv` file dynamically during execution, bypassing the need for an existing text file block. Example dictionary keys:
    - `'initial_lambda'` (float, default 220): Initial lambda [deg].
    - `'initial_lambda_fixed'` (int, default 1): 0/1 fixed/free.
    - `'initial_beta'` (float, default 0): Initial beta [deg].
    - `'initial_beta_fixed'` (int, default 1): 0/1 fixed/free.
    - `'initial_period'` (float, default 5.76198): The starting rotational period estimate in **hours**.
    - `'initial_period_fixed'` (int, default 1): 0/1 fixed/free.
    - `'zero_time'` (float, default 0): Zero time [JD].
    - `'initial_rotation_angle'` (float, default 0): Initial rotation angle [deg].
    - `'convexity_regularization'` (float, default 0.1): Convexity regularization parameter.
    - `'spherical_harmonics_degree'` (int, default 6): Degree of spherical harmonics expansion.
    - `'spherical_harmonics_order'` (int, default 6): Order of spherical harmonics expansion.
    - `'number_of_rows'` (int, default 8): Number of rows.
    - `'phase_func_a'` (float, default 0.5): Phase funct. param. 'a'.
    - `'phase_func_a_fixed'` (int, default 0): 0/1 fixed/free.
    - `'phase_func_d'` (float, default 0.1): Phase funct. param. 'd'.
    - `'phase_func_d_fixed'` (int, default 0): 0/1 fixed/free.
    - `'phase_func_k'` (float, default -0.5): Phase funct. param. 'k'.
    - `'phase_func_k_fixed'` (int, default 0): 0/1 fixed/free.
    - `'lambert_c'` (float, default 0.1): Lambert coefficient 'c'.
    - `'lambert_c_fixed'` (int, default 0): 0/1 fixed/free.
    - `'iteration_stop_condition'` (int, default 50): Iteration stop condition.
-   `conjgradinv_options` (dict, optional): Provides settings to construct the `input_conjgradinv` file required by the underlying Minkowski reconstruction phase. Allowed keys:
    - `'convexity_weight'` (float, default 0.2)
    - `'number_of_rows'` (int, default 8)
    - `'number_of_iterations'` (int, default 100)
    
**(Note: You must provide either an existing `param_file` path OR an `inversion_options` configurations dictionary)**

**Returns:**
-   `vertices` (np.ndarray): A numpy array of shape `(N, 3)` containing the X, Y, Z coordinates of the asteroid's vertices.
-   `faces` (list[list[int]]): A list where each element is a face, defined by a list of 1-based vertex indices.

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
