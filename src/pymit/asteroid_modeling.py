import os
import csv
import subprocess
from pathlib import Path
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
from pathlib import Path

# Module directory paths
MODULE_DIR = Path(__file__).parent.resolve()
# We are currently in src/pymit/, so project root is two levels up
PROJECT_ROOT = MODULE_DIR.parent.parent

DAMIT_DIR = PROJECT_ROOT / 'damit'
CONVEXINV_EXEC = DAMIT_DIR / 'convexinv' / 'convexinv'
MINKOWSKI_EXEC = DAMIT_DIR / 'fortran' / 'minkowski'

class AsteroidModelError(Exception):
    """Exception raised for errors in the Asteroid Modeling pipeline."""
    pass

def run_convexinv(param_file: str, lightcurve_file: str, output_areas_file: str, output_lc_file: str) -> None:
    """
    Run the convexinv binary to generate face areas and normals from light curves.
    
    Args:
        param_file (str): Path to the input parameters file.
        lightcurve_file (str): Path to the light curves input file.
        output_areas_file (str): Path to write the resulting face areas and normals.
        output_lc_file (str): Path to write the modeled light curves.
    """
    if not CONVEXINV_EXEC.exists():
         raise FileNotFoundError(f"convexinv executable not found at {CONVEXINV_EXEC}. Please run 'make' in {CONVEXINV_EXEC.parent}.")

    with open(lightcurve_file, 'rb') as f_in:
        cmd = [
            str(CONVEXINV_EXEC),
            "-o", output_areas_file,
            param_file,
            output_lc_file
        ]
        
        try:
            result = subprocess.run(cmd, stdin=f_in, capture_output=True, text=True, check=True)
            if result.stdout:
                # print(result.stdout)
                pass
        except subprocess.CalledProcessError as e:
            raise AsteroidModelError(f"convexinv failed with return code {e.returncode}.\nStderr: {e.stderr}") from e


def run_minkowski(areas_normals_file: str) -> tuple[np.ndarray, list[list[int]]]:
    """
    Run the minkowski binary to reconstruct a 3D shape from face areas and normals.
    
    Args:
        areas_normals_file (str): Path to the file containing face areas and normals (output from convexinv).
        
    Returns:
        tuple[np.ndarray, list[list[int]]]: 
            - vertices (numpy array of shape (N, 3))
            - faces (list of lists containing 1-based vertex indices for each face)
    """
    if not MINKOWSKI_EXEC.exists():
         raise FileNotFoundError(f"minkowski executable not found at {MINKOWSKI_EXEC}. Please compile 'minkowski.f'.")

    with open(areas_normals_file, 'rb') as f_in:
        try:
            # The minkowski code writes to stdout and uses stdin.
            result = subprocess.run([str(MINKOWSKI_EXEC)], stdin=f_in, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise AsteroidModelError(f"minkowski failed with return code {e.returncode}.\nStderr: {e.stderr}") from e
    
    return _parse_minkowski_output(result.stdout)


def _parse_minkowski_output(output: str) -> tuple[np.ndarray, list[list[int]]]:
    """
    Parse the standard output of the minkowski fortran binary.
    
    Format:
    num_vertices, num_faces
    v1_x v1_y v1_z
    v2_x v2_y v2_z
    ...
    [for each face]
    num_vertices_in_face
    idx1 idx2 idx3 ... (1-based indices)
    """
    lines = output.strip().splitlines()
    if not lines:
        raise ValueError("Empty output from minkowski.")
        
    # First line contains numbers of vertices and facets
    try:
        parts = lines[0].split()
        num_vertices = int(parts[0])
        num_faces = int(parts[1])
    except (ValueError, IndexError) as e:
        raise ValueError(f"Failed to parse header 'num_vertices num_faces' from: {lines[0]}") from e
        
    v_lines = lines[1:num_vertices + 1]
    f_lines = lines[num_vertices + 1:]
    
    # Parse vertices
    vertices = []
    for line in v_lines:
        coords = [float(x) for x in line.split()]
        if len(coords) != 3:
            raise ValueError(f"Invalid vertex line: {line}")
        vertices.append(coords)
        
    # Parse faces
    faces = []
    idx = 0
    while idx < len(f_lines):
        if not f_lines[idx].strip():
            idx += 1
            continue
            
        num_verts_in_face = int(f_lines[idx].strip())
        idx += 1
        
        face_indices_str = f_lines[idx].strip().split()
        face_indices = [int(v) for v in face_indices_str]
        if len(face_indices) != num_verts_in_face:
            raise ValueError(f"Expected {num_verts_in_face} vertices for face, got {len(face_indices)}: {f_lines[idx]}")
            
        faces.append(face_indices)
        idx += 1
        
    return np.array(vertices), faces


def _triangulate_faces(faces: list[list[int]]) -> list[tuple[int, int, int]]:
    """
    Convert arbitrary polygonal faces to triangles (required for some plotting).
    Assumes faces are convex and vertices are given in counter-clockwise order.
    
    Returns 0-based index tuples.
    """
    triangles = []
    for face in faces:
        # Fortran outputs 1-based indices, so convert to 0-based
        face_0 = [idx - 1 for idx in face]
        
        # Simple fan triangulation from the first vertex
        v0 = face_0[0]
        for i in range(1, len(face_0) - 1):
            triangles.append((v0, face_0[i], face_0[i+1]))
            
    return triangles


def load_model_obj(file_path: str) -> tuple[np.ndarray, list[list[int]]]:
    """
    Load an asteroid 3D shape model from a standard Wavefront .obj file.
    
    Args:
        file_path: Path to the .obj file.
        
    Returns:
        vertices: A numpy array of shape (N, 3) where N is the number of vertices.
        faces: A list of lists, where each sub-list contains 1-based vertex indices 
               defining a polygonal face, matching the `minkowski` pipeline standard.
    """
    vertices = []
    faces = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
                
            parts = line.strip().split()
            if parts[0] == 'v':
                # Parse vertex coordinates
                # Format: v x y z
                if len(parts) >= 4:
                    coords = [float(p) for p in parts[1:4]]
                    vertices.append(coords)
            elif parts[0] == 'f':
                # Parse polygonal face indices
                # Format: f v1 v2 v3 ... or f v1/vt1/vn1 v2/vt2/vn2 ...
                face_indices = []
                for p in parts[1:]:
                    # The vertex index is the first element before any slashes
                    v_idx_str = p.split('/')[0]
                    face_indices.append(int(v_idx_str))
                faces.append(face_indices)
                
    return np.array(vertices), faces


def plot_model(vertices: np.ndarray, faces: list[list[int]], save_path: str = None, show: bool = True) -> None:
    """
    Visualize the 3D asteroid shape model.
    
    Args:
        vertices: Numpy array of shape (N, 3) with vertex coordinates.
        faces: List of lists containing 1-based vertex indices for each face.
        save_path: Optional path to save the plot.
        show: Whether to display the plot interactively.
    """
    triangles = _triangulate_faces(faces)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create Poly3DCollection
    # Extract coordinates for each triangle
    tri_coords = vertices[[list(tri) for tri in triangles]]
    
    collection = Poly3DCollection(tri_coords, facecolors='lightgray', linewidths=0.5, edgecolors='k', alpha=0.9)
    ax.add_collection3d(collection)
    
    # Auto-scale axes to fit the model correctly
    all_x = vertices[:, 0]
    all_y = vertices[:, 1]
    all_z = vertices[:, 2]
    
    max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 2.0
    
    mid_x = (all_x.max()+all_x.min()) * 0.5
    mid_y = (all_y.max()+all_y.min()) * 0.5
    mid_z = (all_z.max()+all_z.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Asteroid Shape Model")
    
    # Adjust viewing angle
    ax.view_init(elev=30, azim=45)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if show:
        plt.show()
    
    plt.close()

def save_model_obj(vertices: np.ndarray, faces: list[list[int]], save_path: str) -> None:
    """
    Save the 3D model in Wavefront .obj format.
    
    Args:
        vertices: Numpy array of shape (N, 3) with vertex coordinates.
        faces: List of lists containing 1-based vertex indices.
        save_path: Output .obj file path.
    """
    with open(save_path, 'w') as f:
        f.write("# Asteroid Shape Model\n")
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
        f.write("\n")
        # Write faces (already 1-based indices from Fortran)
        for face in faces:
            # Join indices with spaces
            face_str = " ".join(str(idx) for idx in face)
            f.write(f"f {face_str}\n")


def plot_model_plotly(vertices: np.ndarray, faces: list[list[int]], save_path: str = None, show: bool = True) -> None:
    """
    Visualize the 3D asteroid shape model using Plotly for interactive web viewing.
    
    Args:
        vertices: Numpy array of shape (N, 3) with vertex coordinates.
        faces: List of lists containing 1-based vertex indices.
        save_path: Optional path to save the HTML plot.
        show: Whether to open the plot interactively in the browser.
    """
    # Plotly requires triangles, and uses 0-based indices
    triangles = _triangulate_faces(faces)
    
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    
    i = [tri[0] for tri in triangles]
    j = [tri[1] for tri in triangles]
    k = [tri[2] for tri in triangles]
    
    fig = go.Figure(data=[
        go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color='lightgray',
            opacity=1.0,
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2, roughness=0.5),
            lightposition=dict(x=100, y=100, z=100)
        )
    ])
    
    # Update layout for a better 3D look
    fig.update_layout(
        title="Interactive Asteroid Shape Model",
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='data' # Ensures equal scaling for x, y, z
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    if save_path:
        fig.write_html(save_path)
        
    if show:
        fig.show()

def csv_to_lcs_format(csv_file: str, output_file: str) -> None:
    """
    Convert a standard CSV file containing light curve data into the text format 
    expected by the convexinv executable.
    
    Expected CSV Columns:
    - jd (float)
    - brightness (float)
    - sun_x (float)
    - sun_y (float)
    - sun_z (float)
    - earth_x (float)
    - earth_y (float)
    - earth_z (float)
    Optional columns:
    - curve_id (int/str): To group rows into separate light curves. Default is all same curve.
    - is_relative (int): 0 for absolute, 1 for relative. Default is 0.
    """
    curves = {}
    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            # Parse row
            try:
                jd = float(row['jd'])
                bright = float(row['brightness'])
                sx, sy, sz = float(row['sun_x']), float(row['sun_y']), float(row['sun_z'])
                ex, ey, ez = float(row['earth_x']), float(row['earth_y']), float(row['earth_z'])
            except KeyError as e:
                raise ValueError(f"Missing required column in CSV: {e}")
            except ValueError as e:
                raise ValueError(f"Invalid data format in CSV row {row_idx}: {e}")
                
            cid = row.get('curve_id', '1')
            is_rel = int(row.get('is_relative', '0'))
            
            if cid not in curves:
                curves[cid] = {'is_relative': is_rel, 'points': []}
                
            curves[cid]['points'].append((jd, bright, sx, sy, sz, ex, ey, ez))
            
    with open(output_file, 'w') as f_out:
        # Write total number of curves
        f_out.write(f"{len(curves)}\n")
        
        for cid, data in curves.items():
            pts = data['points']
            # Write num_points, is_relative (note: convexinv expects 0 for relative, 1 for absolute in its internal `1 - i_temp` logic,
            # but wait, the c code has: Inrel[i] = 1 - i_temp ... so if i_temp = 1, Inrel=0 (absolute)
            # The test_lcs_abs file has '81 1' -> 81 points, absolute (1).
            # So is_relative in the csv logic: let's map user 'is_relative (0/1)' to the fort/C format.
            # If user says is_relative=0 (absolute), we should output 1 in the txt file.
            # If user says is_relative=1 (relative), we should output 0 in the txt file.
            out_is_rel = 0 if data['is_relative'] == 1 else 1
            f_out.write(f"{len(pts)} {out_is_rel}\n")
            
            for pt in pts:
                jd, br, sx, sy, sz, ex, ey, ez = pt
                f_out.write(f"{jd:.6f} {br:.6e}  {sx:.6e} {sy:.6e} {sz:.6e}  {ex:.6e} {ey:.6e} {ez:.6e}\n")


def create_convexinv_param_file(options: dict, output_file: str) -> None:
    """
    Programmatically create the convexinv parameter file.
    
    Default options mirror the standard `input_convexinv` typically used.
    
    Args:
        options (dict): A dictionary overriding default inversion settings.
        output_file (str): Path to write the text file payload.
    """
    
    defaults = {
        'initial_lambda': 220,
        'initial_lambda_fixed': 1,
        'initial_beta': 0,
        'initial_beta_fixed': 1,
        'initial_period': 5.76198,
        'initial_period_fixed': 1,
        'zero_time': 0,
        'initial_rotation_angle': 0,
        'convexity_regularization': 0.1,
        'spherical_harmonics_degree': 6,
        'spherical_harmonics_order': 6,
        'number_of_rows': 8,
        'phase_func_a': 0.5,
        'phase_func_a_fixed': 0,
        'phase_func_d': 0.1,
        'phase_func_d_fixed': 0,
        'phase_func_k': -0.5,
        'phase_func_k_fixed': 0,
        'lambert_c': 0.1,
        'lambert_c_fixed': 0,
        'iteration_stop_condition': 50
    }
    
    # Merge user options over defaults
    opts = {**defaults, **(options or {})}
    
    lines = [
        f"{opts['initial_lambda']}\t{opts['initial_lambda_fixed']}\tinital lambda [deg] (0/1 - fixed/free)",
        f"{opts['initial_beta']}\t{opts['initial_beta_fixed']}\tinitial beta [deg] (0/1 - fixed/free)",
        f"{opts['initial_period']}\t{opts['initial_period_fixed']}\tinital period [hours] (0/1 - fixed/free)",
        f"{opts['zero_time']}\t\tzero time [JD]",
        f"{opts['initial_rotation_angle']}\t\tinitial rotation angle [deg]",
        f"{opts['convexity_regularization']}\t\tconvexity regularization",
        f"{opts['spherical_harmonics_degree']} {opts['spherical_harmonics_order']}\t\tdegree and order of spherical harmonics expansion",
        f"{opts['number_of_rows']}\t\tnumber of rows",
        f"{opts['phase_func_a']}\t{opts['phase_func_a_fixed']}\tphase funct. param. 'a' (0/1 - fixed/free)",
        f"{opts['phase_func_d']}\t{opts['phase_func_d_fixed']}\tphase funct. param. 'd' (0/1 - fixed/free)",
        f"{opts['phase_func_k']}\t{opts['phase_func_k_fixed']}\tphase funct. param. 'k' (0/1 - fixed/free)",
        f"{opts['lambert_c']}\t{opts['lambert_c_fixed']}\tLambert coefficient 'c' (0/1 - fixed/free)",
        f"{opts['iteration_stop_condition']}\t\titeration stop condition"
    ]
    
    with open(output_file, 'w') as f:
        f.write("\n".join(lines) + "\n")


def run_pipeline(param_file: str = None, lightcurve_file: str = None, output_areas_file: str = None, output_lc_file: str = None, 
                 plot_file: Union[str, bool] = None, obj_file: Union[str, bool] = None, plotly_file: Union[str, bool] = None, 
                 output_dir: str = None, asteroid_name: str = None, inversion_options: dict = None) -> tuple[np.ndarray, list[list[int]]]:
    """
    Run the full pipeline: compute areas from light curves, reconstruct 3D shape, and visualize.
    
    Args:
        param_file: Path to convexinv input parameters file.
        lightcurve_file: Path to light curves data (txt or csv).
        output_areas_file: Filename to save intermediate areas data. Defaults to '{asteroid_name}_areas.txt'.
        output_lc_file: Filename to save modeled light curves. Defaults to '{asteroid_name}_lcs.txt'.
        plot_file: Name/path for static image plot. Pass True to generate the default '{asteroid_name}_model.png'.
        obj_file: Name/path for exported OBJ model. Pass True to generate the default '{asteroid_name}_model.obj'.
        plotly_file: Name/path for interactive HTML plot. Pass True to generate the default '{asteroid_name}_model.html'.
        output_dir: Directory to save all generated output files. If None, saves to current directory.
        asteroid_name: Optional name prefix for output files. Defaults to 'asteroid'.
        inversion_options: Optional dictionary of overrides to programmatically generate the `input_convexinv` parameter file. If passed, `param_file` will be ignored/overwritten.
    """
    base_name = asteroid_name if asteroid_name else "asteroid"
    
    # Resolve default filenames if not explicitly typed
    if output_areas_file is None:
        output_areas_file = f"{base_name}_areas.txt"
    if output_lc_file is None:
        output_lc_file = f"{base_name}_lcs.txt"
        
    if plot_file is True:
        plot_file = f"{base_name}_model.png"
    elif plot_file is False:
        plot_file = None
        
    if obj_file is True:
        obj_file = f"{base_name}_model.obj"
    elif obj_file is False:
        obj_file = None
        
    if plotly_file is True:
        plotly_file = f"{base_name}_model.html"
    elif plotly_file is False:
        plotly_file = None

    # Set up output directory
    out_dir = Path(output_dir) if output_dir else Path('.')
    if output_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        
    actual_output_areas = str(out_dir / output_areas_file)
    actual_output_lc = str(out_dir / output_lc_file)
    actual_lightcurve_file = lightcurve_file
    
    tmp_lcs_file = None
    tmp_param_file = None
    
    # Handle dictionary configs for convexinv
    if inversion_options is not None:
        print("Generating dynamic input_convexinv settings...")
        tmp_param_file = str(out_dir / f"{base_name}_input_convexinv.txt")
        create_convexinv_param_file(inversion_options, tmp_param_file)
        param_file = tmp_param_file
    elif not param_file:
        raise ValueError("Must provide either a 'param_file' path or 'inversion_options' dict.")
    
    if lightcurve_file.lower().endswith('.csv'):
        print(f"Converting CSV input {lightcurve_file} to convexinv text format...")
        tmp_lcs_file = str(Path(lightcurve_file).with_suffix('.txt'))
        csv_to_lcs_format(lightcurve_file, tmp_lcs_file)
        actual_lightcurve_file = tmp_lcs_file
        
    print("Running convexinv... (this might take a few moments)")
    try:
        run_convexinv(param_file, actual_lightcurve_file, actual_output_areas, actual_output_lc)
        print("convexinv complete.")
    finally:
        # Cleanup temporary files if we created them
        if tmp_lcs_file and Path(tmp_lcs_file).exists() and tmp_lcs_file != lightcurve_file:
            print(f"Cleaning up temporary lightcurves file {tmp_lcs_file}...")
            Path(tmp_lcs_file).unlink()
        if tmp_param_file and Path(tmp_param_file).exists():
            print(f"Cleaning up temporary param file {tmp_param_file}...")
            Path(tmp_param_file).unlink()
            
    print("Running minkowski 3D reconstruction...")
    vertices, faces = run_minkowski(actual_output_areas)
    print(f"Reconstruction complete: {len(vertices)} vertices, {len(faces)} faces.")
    
    if obj_file:
        actual_obj = str(out_dir / obj_file)
        print(f"Saving 3D model to {actual_obj}...")
        save_model_obj(vertices, faces, save_path=actual_obj)
    
    if plot_file:
        actual_plot = str(out_dir / plot_file)
        print(f"Plotting matplotlib model to {actual_plot}...")
        plot_model(vertices, faces, save_path=actual_plot, show=False)
        
    if plotly_file:
        actual_plotly = str(out_dir / plotly_file)
        print(f"Generating interactive plotly model to {actual_plotly}...")
        plot_model_plotly(vertices, faces, save_path=actual_plotly, show=False)
        
    return vertices, faces

if __name__ == "__main__":
    # Example test run if executed directly
    # Assumes input_convexinv and test_lcs_abs exist in the damit folder
    param_txt = DAMIT_DIR / "input_convexinv"
    lcs_txt = DAMIT_DIR / "test_lcs_abs"
    
    if param_txt.exists() and lcs_txt.exists():
        run_pipeline(
            param_file=str(param_txt), 
            lightcurve_file=str(lcs_txt), 
            plot_file=True,
            obj_file=True,
            plotly_file=True,
            output_dir="pipeline_output",
            asteroid_name="test_asteroid"
        )
    else:
        print("Example data files not found. Please provide them to test the pipeline.")
