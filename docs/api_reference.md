# Asteroid Modeling API Reference

This document provides detailed information about the functions available in the `asteroid_modeling` module. For an introduction and installation instructions, see the main [README.md](../README.md).

## Core Functions

### `run_pipeline`
The core function to execute the full end-to-end pipeline.

```python
def run_pipeline(
    param_file: str, 
    lightcurve_file: str, 
    output_areas_file: str = None, 
    output_lc_file: str = None, 
    plot_file: Union[str, bool] = None,
    obj_file: Union[str, bool] = None,
    plotly_file: Union[str, bool] = None,
    output_dir: str = None,
    asteroid_name: str = None,
    inversion_options: dict = None
) -> tuple[np.ndarray, list[list[int]]]:
```

**Arguments:**
-   `param_file` (str): Path to the `input_convexinv` configuration file. This file contains parameters like initial period, regularizations, and iteration limits.
-   `lightcurve_file` (str): Path to the observed light curves data file. **Note:** This can either be in the original plain text format, or it can be a standard `.csv` file. If a `.csv` file is provided, it is automatically converted into the target text format before execution. The expected CSV columns are `jd`, `brightness`, `sun_x`, `sun_y`, `sun_z`, `earth_x`, `earth_y`, `earth_z`, with optional `curve_id` and `is_relative` logic mapping lines into distinct light curves.
-   `output_areas_file` (str, optional): Path where the intermediate output containing areas and normals will be saved. Defaults to `{asteroid_name}_areas.txt` if `asteroid_name` is provided, otherwise `output_areas.txt`.
-   `output_lc_file` (str, optional): Path where the modeled (fitted) light curves will be output. Defaults to `{asteroid_name}_lcs.txt` if string formatting is leveraged, otherwise `output_lcs.txt`.
-   `plot_file` (str or bool, optional): If provided as a string, saves a static matplotlib visualization to this given filename. If `True`, generates a default `{asteroid_name}_model.png`.
-   `obj_file` (str or bool, optional): If provided as a string, saves the exact 3D polygon shape to a standard Wavefront `.obj`. If `True`, generates a default `.obj` file.
-   `plotly_file` (str or bool, optional): If provided as a string, saves an interactive HTML 3D visualization. If `True`, generates a default `.html` template based on the asteroid.
-   `output_dir` (str, optional): A directory where all the specified outputs (`output_areas_file`, `output_lc_file`, plots, and models) will be saved. If it doesn't exist, it will be created. If not provided, it saves to the current working directory.
-   `asteroid_name` (str, optional): Used to dynamically fill output filenames if no strict path is provided. Defaults to "asteroid".
-   `inversion_options` (dict, optional): Overrides individual numerical tuning flags to construct the `input_convexinv` file dynamically during execution, bypassing the need for an existing text file block. Example dictionary keys:
    - `'initial_period'` (float, default 5.76198)
    - `'convexity_regularization'` (float, default 0.1)
    - `'spherical_harmonics_degree'` (int, default 6)
    - `'iteration_stop_condition'` (int, default 50)
    
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
