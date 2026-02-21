# Asteroid Modeling Python Module Documentation

## Overview
The `asteroid_modeling.py` module provides a Python orchestration layer to automate the inversion of asteroid light curves into 3D convex shape models. It wraps two established numerical codes:
1.  **convexinv (C)**: Computes the Gaussian image of the asteroid (face areas and normals) from observed light curves.
2.  **minkowski (Fortran)**: Reconstructs the 3D vertices and polygonal faces from the areas and normals using Minkowski's theorem.

The module also provides a built-in visualization function using `matplotlib` to render the resulting 3D shape.

## Prerequisites
To use this module, you must have the following installed:
-   `numpy`
-   `pandas` (for DataFrame lightcurve support)
-   `matplotlib`
-   `plotly` (for interactive 3D rendering)
-   Compiled executables for `convexinv` and `minkowski` in the `damit` folder. 

- The source code for DAMIT can be downloaded from: [https://damit.cuni.cz/projects/damit/files/version_0.2.1.tar.gz](https://damit.cuni.cz/projects/damit/files/version_0.2.1.tar.gz).

You can install the Python dependencies via pip:
```bash
pip install numpy pandas matplotlib plotly
```

### Quick Links
-   [**Getting Started Guide**](docs/getting_started.md): A step-by-step tutorial on setting up your environment, formatting lightcurves, and running your first inversion.
-   [**API Reference**](docs/api_reference.md): Complete documentation of functions, `inversion_options`, and output parameters.
-   [**Scientific Explanation**](docs/inversion_method.md): A detailed overview of the mathematical theories and inversion techniques driving the module.

## Quick Start Example

If you already have your data prepared and compiled the executables, here's a minimal example:

```python
import pymit
import json

# Instantiate the modeler
modeler = pymit.AsteroidModeler(asteroid_name="Eros", output_dir="data")

# Load lightcurves
modeler.load_lightcurves("test_lcs.csv")

# Configure inversion parameters using a JSON configuration
inv_config = {
    "initial_period": 5.76198
}
conj_config = {
    "number_of_iterations": 150
}
modeler.load_parameters(inversion_json=json.dumps(inv_config), conjgradinv_json=json.dumps(conj_config))

# Run inversion
vertices, faces = modeler.run_inversion()

# Plot and export results
modeler.plot_lightcurves_results(max_curves=3, show=True)
modeler.plot_model(show=True)

print(f"Generated an asteroid model with {len(vertices)} vertices and {len(faces)} faces.")
```

**Modeled output examples:**
![Modeled Light Curves](docs/assets/lightcurves.png)
![Asteroid 3D Shape Model](docs/assets/shape_model.png)

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
