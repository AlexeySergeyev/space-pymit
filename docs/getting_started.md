# Getting Started with pymit

This guide will walk you through the process of setting up the environment, preparing your lightcurve data, and running your very first asteroid shape inversion using the `pymit` module.

## 1. Environment Setup

First, ensure you have the necessary Python dependencies. A `requirements.txt` file is provided in the repository root.

```bash
pip install -r requirements.txt
```

Alternatively, install the core Python packages manually:
```bash
pip install numpy pandas matplotlib plotly requests streamlit
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

The core algorithm requires asteroid lightcurves. While the original `convexinv` program expects a specific plain-text format, `pymit` can load standard `.csv` files, native DAMIT/`convexinv` text files, or DAMIT plaintext export URLs.

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

Here's an example script to run your first inversion from a DAMIT plaintext lightcurve URL:

```python
import pymit

damit_url = (
    "https://damit.cuni.cz/projects/damit/LightCurves/"
    "exportAllForAsteroid/7753/plaintext/A7753.lc.txt"
)

modeler = pymit.AsteroidModeler(asteroid_name="MyAsteroid", output_dir="pipeline_output")
modeler.load_lightcurves(damit_url)

# Configure the inversion step dynamically via a python dictionary
inv_config = {
    "initial_period": 33.0,
    "initial_period_fixed": 1,
    "initial_lambda": 269.0,
    "initial_lambda_fixed": 1,
    "initial_beta": 62.0,
    "initial_beta_fixed": 1,
    # DAMIT LSL scattering parameters p1/p2/p3/p4.
    "phase_func_c": 0.1,
    "phase_func_c_fixed": 0,
    "phase_func_a": 0.5,
    "phase_func_a_fixed": 0,
    "phase_func_d": 0.1,
    "phase_func_d_fixed": 0,
    "phase_func_k": -1.05,
    "phase_func_k_fixed": 0,
    "number_of_rows": 8,
    "iteration_stop_condition": 50,
}
modeler.load_parameters(inversion_json=inv_config)

# Run the complete inversion pipeline
vertices, faces = modeler.run_inversion()

# Plot and generate outputs in output_dir.
# The observed-vs-modeled lightcurve plot uses solar phase angle from the lightcurve vectors.
modeler.plot_lightcurves_results(save=True, show=False)
modeler.plot_model(save=True, show=False)
modeler.plot_model_plotly(save=True, show=False)
modeler.export_obj()

# Optional: compute sky projection and a synthetic lightcurve at a given Julian date.
# The modeler uses the fitted period and the first observation JD as t0.
modeler.plot_sky_projection(save=True, show=False, jd=2461169.0)
modeler.plot_synthetic_lightcurve(save=True, show=False, jd=2461169.0)

print(f"Success! Reconstructed an asteroid shape with {len(vertices)} vertices and {len(faces)} faces.")
```

When you run this script:
1.  **Input Preparation**: CSV inputs are converted automatically; native DAMIT text files and DAMIT URLs are passed to `convexinv` in their original format.
2.  **Inversion**: The `convexinv` C executable computes the Gaussian image (areas and normals).
3.  **Reconstruction**: The `minkowski` Fortran executable computes the precise 3D geometry (vertices, faces).
4.  **Outputs**: A 3D `.obj` model, Matplotlib `.png` plots, an observed-vs-modeled phase curve, an interactive `.html` plot, and optional projection/lightcurve `.csv` files will be saved to the `pipeline_output/` folder.

---

## 4. Running the Local GUI

You can also run the modeling workflow through a local Streamlit interface.

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Start the GUI from the repository root:

```bash
PYTHONPATH=src streamlit run apps/streamlit_app.py
```

The GUI supports:
- DAMIT plaintext lightcurve URLs
- Uploaded `.csv` lightcurve files
- Uploaded native DAMIT/`convexinv` text files
- Inversion and Minkowski parameter controls
- DAMIT LSL scattering controls: `p1/c`, `p2/a`, `p3/d`, and `p4/k`
- Sky projection and synthetic lightcurve outputs for a selected Julian date
- Interactive 3D shape display
- Download buttons for generated OBJ, HTML, PNG, TXT, and CSV outputs

Sidebar values are saved to `.pymit_last_params.json` in the repository root after a successful run and restored the next time the GUI starts.

---

## Next Steps

- Check out the [API Reference](api_reference.md) for a comprehensive list of all `inversion_options` and function arguments.
- Explore the output files in your specified `output_dir` to see the intermediate `_areas.txt` and `_lcs.txt` text outputs.
