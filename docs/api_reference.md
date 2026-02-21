# Asteroid Modeling API Reference

This document provides detailed information about the functions available in the `asteroid_modeling` module. For an introduction and installation instructions, see the main [README.md](../README.md).

## Core Classes

### `AsteroidModeler`
The main class to handle data state, configuration parameters, and the execution of the full end-to-end pipeline.

```python
class AsteroidModeler:
    def __init__(self, asteroid_name: str = "asteroid", output_dir: str = "."):
```
Initializes the modeler instance.
-   `asteroid_name` (str): The name used as a prefix for saved output files. Defaults to "asteroid".
-   `output_dir` (str): Directory where outputs are saved. Created if it doesn't exist. Defaults to current directory.

#### `load_lightcurves`
```python
    def load_lightcurves(self, source: Union[str, pd.DataFrame]) -> None:
```
Loads lightcurves into the modeler.
-   `source` (str or pd.DataFrame): Path to the observed light curves data file (`.csv` or `.txt` LCS format) OR a `pandas.DataFrame`.

#### `load_parameters`
```python
    def load_parameters(
        self,
        inversion_json: Union[str, dict] = None,
        conjgradinv_json: Union[str, dict] = None
    ) -> None:
```
Loads inversion and shape construction options.
-   `inversion_json` (str or dict, optional): Configuration for the convex inversion step (`input_convexinv`). Can be a JSON file path, JSON string, or a dictionary. 
    Examples of keys: `'initial_period'`, `'convexity_regularization'`, `'iteration_stop_condition'`.
-   `conjgradinv_json` (str or dict, optional): Configuration for the shape reconstruction step (`input_conjgradinv`).

#### `run_inversion`
```python
    def run_inversion(self) -> tuple[np.ndarray, list[list[int]]]:
```
Executes the shape reconstruction using the loaded lightcurves and parameters.
**Returns:**
-   `vertices` (np.ndarray): A numpy array of shape `(N, 3)` containing the X, Y, Z coordinates.
-   `faces` (list[list[int]]): A list where each element is a face, defined by a list of 1-based vertex indices.

#### Shape and Plot Utils (Object Methods)
-   `plot_lightcurves_results(max_curves=3, show=True, save_path=None)`: Plots observed vs modeled lightcurves.
-   `plot_model(show=True, save_path=None)`: Renders the 3D shape using matplotlib.
-   `plot_model_plotly(show=True, save_path=None)`: Renders the 3D shape as an interactive HTML file.
-   `export_obj(file_path=None)`: Exports the shape to a `.obj` file.

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
