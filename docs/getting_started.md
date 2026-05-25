# Getting Started With PyMit

PyMit can be used from the Streamlit app or directly from Python. For exploratory local work, start with Streamlit; it exposes the current pipeline controls and keeps generated files downloadable from the browser.

## 1. Install Dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

The core dependencies are `numpy`, `pandas`, `matplotlib`, `plotly`, `requests`, and `streamlit`.

## 2. Compile DAMIT Executables

PyMit calls numerical DAMIT programs written in C and Fortran. Compile them before running either the GUI or Python API:

```bash
cd damit/convexinv
make

cd ../fortran
gfortran minkowski.f -o minkowski
gfortran standardtri.f -o standardtri
```

Expected executables:

- `damit/convexinv/convexinv`
- `damit/convexinv/period_scan`
- `damit/fortran/minkowski`
- `damit/fortran/standardtri`

## 3. Run The Streamlit App

```bash
PYTHONPATH=src streamlit run apps/streamlit_app.py
```

Use the sidebar to choose a DAMIT plaintext URL or upload a CSV/native DAMIT lightcurve file. Then select the pipeline stages:

- **Period scan**: optional pre-scan for the initial period. Saves raw, CSV, and PNG outputs.
- **Convex inversion**: runs `convexinv` with the selected spin, period, shape, and LSL scattering settings.
- **Minkowski reconstruction**: reconstructs the 3D shape from the latest areas file.
- **Pole grid search**: optional golden-spiral scan of initial pole guesses; selects the successful minimum chi-square candidate.
- **Sky projection**: optional projection and synthetic lightcurve products for a selected Julian date.

The app runs the pipeline in a background thread and updates the log while the job is running. After completion, it renders the 3D model, observed-vs-modeled phase curve, folded residuals, period scan, pole-grid maps, projection products, and download buttons for generated files.

## 4. Prepare Lightcurve Data

PyMit accepts:

- DAMIT plaintext lightcurve export URLs
- Native DAMIT/`convexinv` text files
- CSV files
- `pandas.DataFrame` objects

CSV input must include:

- `jd`: Julian date of observation
- `brightness`: observed brightness
- `sun_x`, `sun_y`, `sun_z`: Sun vector coordinates
- `earth_x`, `earth_y`, `earth_z`: Earth vector coordinates

Optional CSV columns:

- `curve_id`: groups rows into individual lightcurves; defaults to `1`
- `is_relative`: `0` for absolute photometry and `1` for relative; defaults to `0`

Native DAMIT text inputs are normalized before `convexinv` when curve header counts do not match the actual number of data rows. Repaired files are written as `*_convexinv.txt` in the output directory.

## 5. Run From Python

```python
import pymit

damit_url = (
    "https://damit.cuni.cz/projects/damit/LightCurves/"
    "exportAllForAsteroid/7753/plaintext/A7753.lc.txt"
)

modeler = pymit.AsteroidModeler(
    asteroid_name="A7753",
    output_dir="pipeline_output",
)
modeler.load_lightcurves(damit_url)
modeler.load_parameters(
    inversion_json={
        "initial_period": 33.0,
        "initial_period_fixed": 1,
        "initial_lambda": 269.0,
        "initial_lambda_fixed": 1,
        "initial_beta": 62.0,
        "initial_beta_fixed": 1,
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
    },
    conjgradinv_json={
        "number_of_iterations": 100,
    },
)

vertices, faces = modeler.run_inversion(
    run_period_scan=True,
    period_scan_options={
        "period_start": 32.0,
        "period_end": 34.0,
        "period_interval_coefficient": 0.8,
        "number_of_rows": 4,
        "minimum_number_of_iterations": 10,
    },
    period_scan_workers=1,
)

modeler.plot_lightcurves_results(save=True, show=False, max_curves=1000)
modeler.plot_model(save=True, show=False)
modeler.plot_model_plotly(save=True, show=False)
modeler.plot_sky_projection(save=True, show=False, jd=2461169.0)
modeler.plot_synthetic_lightcurve(save=True, show=False, jd=2461169.0)
modeler.export_obj()
```

To use a pole grid instead of a single initial pole:

```python
vertices, faces = modeler.run_pole_grid_scan(n=10, workers=1)
```

The golden-spiral grid tests `2N + 1` initial poles. `lambda` is the grid longitude, `beta` is the grid latitude, and the selected candidate is the successful run with the smallest chi-square.

## 6. Inspect Outputs

For `asteroid_name="A7753"`, typical files in `output_dir` are:

- `A7753_period_scan.txt`, `A7753_period_scan.csv`, `A7753_period_scan.png`
- `A7753_input_convexinv.txt`
- `A7753_areas.txt`
- `A7753_lc_output.csv`
- `A7753_lightcurves.html`
- `A7753_lightcurves_folded_residuals.png`
- `A7753_model.png`, `A7753_model.html`, `A7753.obj`
- `A7753_pole_scan_results.csv`, `A7753_pole_scan_best.json`, and pole-map PNGs when pole grid scan is enabled
- `A7753_sky_projection.csv`, `A7753_sky_projection.png`
- `A7753_synthetic_lightcurve.csv`, `A7753_synthetic_lightcurve.png`

## Next Steps

- Read the [API Reference](api_reference.md) for method signatures and helper functions.
- Read the [DAMIT Parameters Guide](damit_parameters.md) before changing shape resolution, convexity weights, or LSL scattering fit switches.
