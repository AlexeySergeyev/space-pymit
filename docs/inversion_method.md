# Scientific Explanation of the Lightcurve Inversion Method

This document provides a theoretical overview of the numerical inversion techniques used to derive 3D asteroid shape, spin, and scattering models from photometric lightcurves. The methodology relies on algorithms originally developed by Mikko Kaasalainen and Josef Ďurech, encapsulated in the DAMIT codebase.

## 1. The Inversion Problem
The core objective of lightcurve inversion is to find a set of physical parameters (spin axis, rotation period, 3D shape, and surface scattering properties) that reproduce a set of observed photometric lightcurves. 

The procedure minimizes the relative $\chi^2$ difference between the observed brightness $L^{(i)}_{\text{obs}}$ and the modeled brightness $L^{(i)}$, leveraging the Levenberg-Marquardt optimization algorithm. 

## 2. Shape Representation: The Gaussian Image
Instead of directly solving for 3D vertex coordinates, the `convexinv` routine optimizes the **Gaussian image** of the asteroid. The Gaussian image is a mathematical representation of a convex polyhedron defined strictly by the **surface areas** of its facets and their **outward-pointing unit normals**.

### Spherical Harmonics Expansion
To reduce the dimensionality of the parameter space, the facet areas are characterized by a Laplace series expansion (Spherical Harmonics) of degree $l$ and order $m$. Typically, $l=m=6$ provides a good balance between resolution and parameter count (resulting in $49$ shape parameters).

### Convexity Regularization and the "Dark Facet"
A purely mathematical optimization of area parameters can produce non-physical geometries that do not close into a convex volume. To enforce formal convexity, a regularization weight is applied alongside a theoretical "dark facet". This facet is necessary to close the geometric representation mathematically but does not contribute photometrically to the lightcurve.

## 3. Light Scattering Model
The brightness of each facet depends on its orientation towards the Sun and the observer. The inversion uses a combined empirical scattering law:

$$S(\mu, \mu_0, \alpha) = f(\alpha) [S_{LS}(\mu, \mu_0) + c S_L(\mu, \mu_0)]$$

Where:
- $\alpha$ is the solar phase angle.
- $\mu_0$ and $\mu$ are the cosines of the angles of incidence and reflection, respectively.
- $S_L = \mu\mu_0$ is **Lambert's law** (diffuse scattering).
- $S_{LS} = \frac{\mu\mu_0}{\mu + \mu_0}$ is the **Lommel-Seeliger law** (single-scattering in particulate media).
- $c$ is the Lambertian weighting coefficient.

The phase function $f(\alpha)$ is often modeled as an exponential-linear form:
$f(\alpha) = a \exp\left(-\frac{\alpha}{d}\right) + k\alpha + 1$

*(Note: For relative lightcurves where absolute magnitude isn't known, phase function parameters are often kept fixed, as optimizing them leads to divergence.)*

## 4. 3D Shape Reconstruction (Minkowski Problem)
Once `convexinv` successfully isolates the optimal Gaussian image (facet areas and fixed normals), the abstract representation must be converted back into a 3D mesh.

This is a classic mathematical challenge known as the **Minkowski Problem**. The `minkowski` Fortran subroutine iteratively solves this problem, directly calculating the 3D vertex coordinates and polygonal face indices that form the unique convex polyhedron corresponding to the optimized facet areas.

## 5. Shape Polishing (Conjugate Gradient Method)
An optional final refinement step uses the `conjgradinv` program. While `convexinv` optimizes the spherical harmonic coefficients to find the shape, `conjgradinv` uses the simpler **conjugate gradient method** to directly optimize the individual facet areas. 

During this polishing phase, the rotation and scattering parameters are kept fixed to the best-fit values found by `convexinv`. Because `conjgradinv` operates directly on surface areas rather than spherical harmonics, it often yields visually "nicer" shapes and slightly tighter statistical fits.

## 6. Period Scanning
Because lightcurves are highly periodic and complex, the $\chi^2$ parameter space contains many local minima. To avoid converging on a false solution, the entire range of plausible rotation periods must be systematically scanned before running the full `convexinv` optimization. The `period_scan` tool evaluates multiple initial pole orientations at small period increments to identify the true global minimum.
