# Asteroid Modeling Module Walkthrough

## What Was Accomplished
I have created a Python orchestration module (`asteroid_modeling.py`) that successfully interfaces with the C `convexinv` and Fortran `minkowski` codebase originally provided in the `docs` directory. 

The pipeline performs the following steps automatically:
1.  **[NEW]** Option to dynamically ingest standardized `.csv` lightcurve formats and build the expected Fortran representations on-the-fly (`csv_to_lcs_format`).
2.  Executes `convexinv` on a set of light curves to determine the Gaussian image (areas and normal vectors) of the target asteroid.
3.  Passes this unformatted output into the Fortran `minkowski` reconstruction engine to solve for the exact 3D shape (vertices and polygonal faces).
4.  Calculates and renders a comprehensive 3D plot visualizer of the asteroid using `matplotlib`.
5.  Optionally exports the exact 3D shape into a `.obj` model file.
6.  Optionally saves an interactive HTML 3D visualization using `plotly`.
7.  **[NEW]** The entire suite of file exports (`output_areas.txt`, `output_lcs.txt`, `.png`, `.html`, `.obj`) can now be targeted to a custom output directory via the `output_dir` parameter.
8.  **[NEW]** Load and analyze existing models using the `load_model_obj` extraction method, bypassing the pipeline.

In addition to writing the Python driver, I have also provided comprehensive markdown documentation (`README.md` in the new `docs/` repository path), featuring detailed guides on data injection formatting and the requisite citation structures dictated by the DAMIT project authors.

## Verification Results
I compiled the C code (using `make`) and the Fortran code (using `gfortran`) inside their respective documentation folders. Next, I wrote the python script and executed it against the provided sample data (`test_lcs_abs` and `input_convexinv`) to ensure it worked end-to-end.

The test run completed successfully:
- The Fortran script cleanly rebuilt `1021` spatial vertices and `513` faces.
- The `matplotlib` engine produced a 3D visualization.
- No memory faults or process errors were encountered during the wrap. 

**Below is the generated 3D shape model plot of the test asteroid:**

![Asteroid Model](/Users/alexeysergeyev/.gemini/antigravity/brain/1ff5dfd9-071a-46b7-b690-4dd501548d02/asteroid_model.png)
