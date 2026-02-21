# Asteroid Modeling Python Module Documentation

## Overview
The `asteroid_modeling.py` module provides a Python orchestration layer to automate the inversion of asteroid light curves into 3D convex shape models. It wraps two established numerical codes:
1.  **convexinv (C)**: Computes the Gaussian image of the asteroid (face areas and normals) from observed light curves.
2.  **minkowski (Fortran)**: Reconstructs the 3D vertices and polygonal faces from the areas and normals using Minkowski's theorem.

The module also provides a built-in visualization function using `matplotlib` to render the resulting 3D shape.

## Prerequisites
To use this module, you must have the following installed:
-   `numpy`
-   `matplotlib`
-   `plotly` (for interactive 3D rendering)
-   Compiled executables for `convexinv` and `minkowski` in the `damit` folder. 

- The source code for DAMIT can be downloaded from: [https://damit.cuni.cz/projects/damit/files/version_0.2.1.tar.gz](https://damit.cuni.cz/projects/damit/files/version_0.2.1.tar.gz).

You can install the Python dependencies via pip:
```bash
pip install numpy matplotlib plotly
```

### Module Context
The directory is arranged as a standard Python module (due to `__init__.py`). Therefore, if the `pymit/` parent directory is in your `PYTHONPATH` or you are executing a script side-by-side with the module folder, you can seamlessly import the full pipeline:

```python
# Import specific functions
from pymit.asteroid_modeling import run_pipeline, load_model_obj, plot_model

# Or import the whole wrapper
import pymit
```

To compile the C and Fortran code, navigate to the `damit/convexinv` and `damit/fortran` directories and run:
```bash
# In damit/convexinv
make

# In damit/fortran
gfortran minkowski.f -o minkowski
gfortran standardtri.f -o standardtri
```

## API Reference

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
    asteroid_name: str = None
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

**Returns:**
-   `vertices` (np.ndarray): A numpy array of shape `(N, 3)` containing the X, Y, Z coordinates of the asteroid's vertices.
-   `faces` (list[list[int]]): A list where each element is a face, defined by a list of 1-based vertex indices.

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

## Example Usage
```python
import asteroid_modeling

param_txt = "damit/input_convexinv"
# The input lightcurves can be custom .txt or a parsed .csv file
lcs_txt = "test_lcs.csv"

# Run the complete inversion pipeline, leveraging automation
# Booleans signal that standard `{asteroid_name}_[output_type]` 
# names should be generated and placed in output_dir.
vertices, faces = asteroid_modeling.run_pipeline(
    param_file=param_txt, 
    lightcurve_file=lcs_txt, 
    plot_file=True,
    obj_file=True,
    plotly_file=True,
    output_dir="pipeline_output",
    asteroid_name="Eros"
)

print(f"Generated an asteroid model with {len(vertices)} vertices and {len(faces)} faces.")
```

## Error Handling
The module raises `AsteroidModelError` if the underlying C/Fortran binaries fail or return a non-zero exit code. Ensure that your input parametrization matches the formatting expectations of the Kaasalainen & Torppa inversion codes.

## Licensing and Citations

The `damit` components used by this module are derived from the [Database of Asteroid Models from Inversion Techniques (DAMIT)](https://damit.cuni.cz/projects/damit/). 

Except where otherwise stated by the authors, content is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). 

**If you use this module and its underlying DAMIT executables for research, please abide by the project's rules:**

1.  **Always cite the original paper where a given model was published.** Give credit to those who derived the models you are using.
2.  Also, cite the core project paper:
    > *Ďurech et al. (2010), DAMIT: a database of asteroid models*, A&A, 513, A46
    > (ADS: [2010A&A...513A..46D](https://ui.adsabs.harvard.edu/abs/2010A%26A...513A..46D))
3.  Provide a link back to the DAMIT website: `https://damit.cuni.cz/projects/damit/`

For non-scientific work, simply providing a link to the website above is sufficient.
