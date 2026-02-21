# DAMIT Parameters Guide: Impact on Modeling, Precision, and Performance

This document describes the underlying configuration parameters used by the DAMIT modeling tools (`convexinv` and `conjgradinv`), which are wrapped by the `pymit` library. Understanding these parameters is crucial for ensuring physically plausible 3D asteroid shape reconstruction, high precision, and optimal convergence performance.

---

## 1. `convexinv` Parameters
The `convexinv` executable computes a shape, spin, and scattering model that gives the best fit to the input lightcurves. It represents the shape as the Gaussian image of a convex polyhedron.

### **Initial Spin and Period (`initial_pole_lambda`, `initial_pole_beta`, `initial_period`)**
*   **Description**: The asteroid's initial ecliptic pole coordinates $\lambda, \beta$ (in degrees) and rotation period $P$ (in hours).
*   **Impact on Modeling**: Defines the starting point for the Levenberg-Marquardt optimizer.
*   **Precision/Performance**: Since `convexinv` finds a local minimum of $\chi^2$, you must supply an `initial_period` close to the global minimum (often found via a period scan). Incorrect initial periods will lead to divergent or erroneous shape models.

### **Convexity Regularization Weight (`convexity_weight`)**
*   **Description**: A weight factor used to keep the resulting shape formally convex. The model utilizes a "dark facet" area that artificially ensures the collection of facets remains convex.
*   **Impact on Modeling**: It strongly affects the overall shape. The dark facet is not photometrically active but maintains mathematical convexity.
*   **Precision/Performance**: You should adjust this parameter to keep the dark facet area below 1% of the total area. 
    *   **Typical value**: `0.1` (can shrink or grow by orders of magnitude).
    *   **Heuristic**: If increasing `convexity_weight` reduces the dark area but worsens the fit significantly, it may indicate a genuine albedo variation over the asteroid's surface that is being masked as a shape deformation.

### **Laplace Series Expansion (`laplace_degree_l`, `laplace_order_m`)**
*   **Description**: The degree $l$ and order $m$ for the spherical harmonics expansion used to estimate the Gaussian image before generating facets. 
*   **Impact on Modeling**: Controls the maximum topological complexity of the shape representation. The total number of shape parameters equals $(l + 1)^2$.
*   **Precision/Performance**: 
    *   **Typical value**: $l=m=6$. 
    *   Higher values allow for more complex shapes but run slower and may introduce overfitting or instability if the data does not constrain the features tightly enough.

### **Resolution (`resolution_n`)**
*   **Description**: The number of triangulation rows $n$ per octant. The total number of model facets (surface areas of the Gaussian image) is $8n^2$.
*   **Impact on Modeling**: Controls the polygon count and geometric smoothness of the reconstructed 3D shape mapped by `minkowski`.
*   **Precision/Performance**: 
    *   **Typical value**: `8` to `10` (corresponding to 512–800 facets).
    *   Running with high values of $n$ increases precision slightly but dramatically inflates `minkowski`'s processing time.

### **Light Scattering Parameters (`scattering_a`, `d`, `k`, `c`)**
*   **Description**: Defines the exponential-linear phase function and a mix of Lommel-Seeliger and Lambert scattering laws.
*   **Impact on Modeling**: Crucial for accurately modeling absolute/calibrated lightcurves containing phase angle variations.
*   **Precision/Performance**:
    *   If relying **only** on relative lightcurves, no solar phase modeling is needed. Attempting to fit these parameters on relative data often causes the algorithm to diverge. In these cases, use fixed parameters with zero amplitude/slope (e.g., $a=0, d=0, k=0, c=0.1$).

### **Hapke Scattering Model Parameters (`w`, `g`, `B0`, `h`, `theta`)**
*   **Description**: In cases where highly accurate, calibrated photometry covers a broad range of phase angles, the more sophisticated Hapke scattering model may be used.
*   **Parameters**:
    *   `w`: Single-particle scattering albedo.
    *   `g`: Asymmetry parameter of the single-particle function.
    *   `B0`: Opposition surge amplitude.
    *   `h`: Opposition surge width.
    *   `theta`: Macroscopic roughness ($\bar{\theta}$).
*   **Impact on Modeling**: Allows for advanced calibration and physical representation of the asteroid's surface reflectiveness structure but requires very high-quality datasets to perform precision fitting.

### **Iteration Stop Condition (`stop_condition`)**
*   **Description**: Defines when the optimization loop ends.
*   **Precision/Performance**: 
    *   If `> 1`: Functions as a strict limit on the number of iterations. Best for predictable performance.
    *   If `< 1`: Represents the delta/difference in the RMS deviation between two bounds. Computation dynamically stops once improvements fall beneath this threshold, prioritizing precision over performance matching.

---

## 2. `conjgradinv` Parameters

The `conjgradinv` program acts as an analogy to `convexinv` but relies on a Conjugate Gradient method. Crucially, rotation and scattering parameters are **fixed**. It uses the direct facet areas natively as its optimization parameters rather than adjusting through spherical harmonics. 

### **Role in the Pipeline**
*   **Purpose**: A “polishing” tool used *after* a successful `convexinv` run. It refines the final shape while inheriting the pole and period.
*   **Impact on Modeling**: Since it sidesteps the Laplace series approximation from `convexinv` and targets area distributions directly, it usually yields aesthetically smoother, "nicer" shapes with marginally tighter fits.

### **Configurable Aspects (`resolution`, `max_iterations`, `convexity_weight`)**
These parameters behave exactly identically to those in `convexinv`.
*   **Resolution and Iterations**: Directly control execution time. Typical configurations pair roughly `n=10` with exactly `100` maximum conjugate gradient steps. 
*   **Convexity Weight**: Continues to manage the dark facet proportion natively within the gradient bounds to enforce valid topological representation.

---

## 3. How to Improve Modeling: Parameter Tuning Guide

To achieve the best possible asteroid shape model, you will often need to iteratively tune the parameters based on the output of previous runs. Here is a practical guide on how to improve your modeling results:

### **Step 1: Finding the Global Period Minimum**
Before tuning any shape parameters, you must ensure your `initial_period` is correct. `convexinv` finds a local minimum. If initialized with the wrong period, the resulting shape will be nonsensical regardless of other parameters.
*   **Action**: Always run a period scan over a wide interval to find the absolute minimum in the $\chi^2$ vs. period plot.
*   **Sign of Failure**: High RMS deviation, or multiple poles giving the same high residual. This indicates you might be trapped in a local minimum or lack sufficient data.

### **Step 2: Controlling the Dark Facet (Convexity)**
The "dark facet" is an artificial construct to force the polyhedron to remain convex. A mathematically sound convex model should have a minimal dark facet.
*   **Action**: Tune the `convexity_weight`.
    *   Start with `0.1`.
    *   If the dark facet area in the output is $>1\%$, **increase** the `convexity_weight` by an order of magnitude (e.g., `1.0`, `10.0`).
*   **Sign of Failure**: If increasing the weight shrinks the dark facet but severely worsens the RMS fit (the lightcurve matching degrades), the asteroid likely has profound albedo variations or large, unmodelable concavities that the optimizer is trying to "fake" using the dark facet.

### **Step 3: Handling Scattering Parameters vs. Data Types**
Using the wrong scattering setup for your data will cause the optimization to diverge.
*   **Action for Relative Data Only**: If all your lightcurves are relative (meaning they are not absolutely calibrated to a standard magnitude system), **do not** attempt to fit scattering parameters $a, d, k$. Fix them to `0` and fix $c$ to `0.1`. 
*   **Action for Calibrated Data**: Allow $a, d, k$ to vary freely. If you have exceptional precision across many phase angles, consider switching to the Hapke parameters to capture the opposition surge accurately.
*   **Sign of Failure**: If `convexinv` fails to converge or outputs mathematical errors, you may be trying to fit scattering parameters to relative data that lacks solar phase angle information.

### **Step 4: Increasing Shape Resolution Carefully**
Once you have a stable model with a good period and a small dark facet, you can attempt to capture finer surface details.
*   **Action**: Increase `resolution_n` from `8` to `10` or `12`. Also increase `laplace_degree_l` and `laplace_order_m` symmetrically (e.g., from `6` to `8`).
*   **Trade-off**: High resolution drastically increases the geometric processing time in `minkowski` and the optimization time in `convexinv`. 
*   **Sign of Failure**: If an increased resolution introduces strange, sharp structural artifacts ("spikes" or extreme flattening) without lowering the $\chi^2$ value, the model is overfitting the noise in your lightcurves. Revert to a lower resolution.

### **Step 5: Final Polish Using `conjgradinv`**
Do not use `conjgradinv` until you are perfectly satisfied with the `convexinv` result.
*   **Action**: Take the fitted rotation parameters from `convexinv` and pass them as fixed inputs to `conjgradinv`. This will refine the facet areas directly without the spherical harmonics approximation, often smoothing out the shape and slightly improving the fit.
