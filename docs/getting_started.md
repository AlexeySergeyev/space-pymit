# Getting Started with pymit

This guide will walk you through the process of setting up the environment, preparing your lightcurve data, and running your very first asteroid shape inversion using the `pymit` module.

## 1. Environment Setup

First, ensure you have the necessary Python dependencies. A `requirements.txt` file is provided in the repository root.

```bash
pip install -r requirements.txt
```

Alternatively, install them manually:
```bash
pip install numpy matplotlib plotly
```

### Compiling Core Executables

This module relies on numerical codes written in C and Fortran from the DAMIT project. You must compile them before using the Python wrappers.

Navigate to the respective directories and compile:

```bash
# Compile convexinv (C)
cd damit/convexinv
make

# Compile minkowski (Fortran)
cd ../fortran
gfortran minkowski.f -o minkowski
gfortran standardtri.f -o standardtri
```

Make sure the resulting executables (`convexinv`, `minkowski`, `standardtri`) are present in their respective folders.

---

## 2. Preparing Lightcurve Data

The core algorithm requires asteroid lightcurves. While the original `convexinv` program expects a specific plain-text format, `pymit` allows you to directly pass standard `.csv` files.

Your `.csv` file should contain the following columns:
- `jd`: Julian date of observation
- `brightness`: Observed brightness
- `sun_x`, `sun_y`, `sun_z`: Sun vector coordinates
- `earth_x`, `earth_y`, `earth_z`: Earth vector coordinates

**Optional columns:**
- `curve_id`: Used to group rows belonging to the same lightcurve (defaults to `1`)
- `is_relative`: `0` for absolute photometry, `1` for relative (defaults to `0`)

---

## 3. Running the Pipeline

You can invoke the pipeline programmatically from any Python script. Ensure that the `src` folder is in your `PYTHONPATH`, or that you install `pymit` as a package.

Here's an example script to run your first inversion:

```python
import pymit

# Path to your CSV file containing lightcurves
lcs_csv = "my_lightcurves.csv"

# Run the complete inversion pipeline
vertices, faces = pymit.run_pipeline(
    param_file=None, # We'll use programmatic options instead of a text file
    lightcurve_file=lcs_csv, 
    inversion_options={
        'initial_period': 5.76198, # Provide initial period estimate in hours
        'convexity_regularization': 0.1,
        'spherical_harmonics_degree': 6,
        'iteration_stop_condition': 25
    },
    plot_file=True,     # Generate a static 3D plot image
    obj_file=True,      # Export a 3D .obj model
    plotly_file=True,   # Export an interactive HTML 3D visualization
    output_dir="pipeline_output", # Folder to store generated files
    asteroid_name="MyAsteroid"    # Prefix for generated files
)

print(f"Success! Reconstructed an asteroid shape with {len(vertices)} vertices and {len(faces)} faces.")
```

When you run this script:
1.  **Format Conversion**: The CSV is automatically converted into the format required by `convexinv`.
2.  **Inversion**: The `convexinv` C executable computes the Gaussian image (areas and normals).
3.  **Reconstruction**: The `minkowski` Fortran executable computes the precise 3D geometry (vertices, faces).
4.  **Outputs**: A 3D `.obj` model, a matplotlib `.png` plot, and an interactive `.html` plot will be saved to the `pipeline_output/` folder.

---

## Next Steps

- Check out the [API Reference](api_reference.md) for a comprehensive list of all `inversion_options` and function arguments.
- Explore the output files in your specified `output_dir` to see the intermediate `_areas.txt` and `_lcs.txt` text outputs.
