import math
import signal
import subprocess
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .errors import AsteroidModelError
from .paths import CONVEXINV_EXEC, MINKOWSKI_EXEC
from .shape import _parse_minkowski_output


_MINKOWSKI_MAX_FACES = 6000
_MINKOWSKI_MAX_AREA_RATIO = 1_000_000.0


def run_convexinv(
    param_file: str,
    lightcurve_file: str,
    output_areas_file: str,
    output_lc_file: str,
    verbose: bool = False,
) -> dict:
    """
    Run the convexinv binary to generate face areas and normals from light curves.
    """
    if not CONVEXINV_EXEC.exists():
        raise FileNotFoundError(
            f"convexinv executable not found at {CONVEXINV_EXEC}. Please run 'make' in {CONVEXINV_EXEC.parent}."
        )

    cmd = [
        str(CONVEXINV_EXEC),
        "-v",
        "-o",
        output_areas_file,
        param_file,
        output_lc_file,
    ]

    collected_stdout = []
    with open(lightcurve_file, "rb") as f_in:
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=f_in,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert proc.stdout is not None and proc.stderr is not None
            for line in proc.stdout:
                collected_stdout.append(line)
                if verbose and not line.startswith("f<0"):
                    print(line, end="", flush=True)
            proc.wait()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(
                    proc.returncode, cmd, stderr=proc.stderr.read()
                )
        except subprocess.CalledProcessError as e:
            raise AsteroidModelError(
                _format_convexinv_failure(
                    e.returncode, e.stderr or "", param_file
                )
            ) from e

    result_stdout = "".join(collected_stdout)

    fit = {}
    for line in result_stdout.splitlines():
        if line.startswith("lambda, beta and period"):
            parts = line.split(":")[1].split()
            fit["lambda"] = float(parts[0])
            fit["beta"] = float(parts[1])
            fit["period"] = float(parts[2])
        elif line.startswith("phase function parameters:"):
            vals = line.split(":")[1].split()
            fit["phase_a"] = float(vals[0])
            fit["phase_d"] = float(vals[1])
            fit["phase_k"] = float(vals[2])
        elif line.startswith("Lambert coefficient:"):
            fit["lambert_c"] = float(line.split(":")[1])
    return fit


def _format_convexinv_failure(returncode: int, stderr: str, param_file: str) -> str:
    if "Singular Matrix" in stderr:
        return (
            f"convexinv failed because its linear solve became singular while using {param_file}.\n"
            "This often happens when too many parameters are free for the available lightcurve data. "
            "For A7753, keep the Lambert coefficient fixed: set phase_func_c to the found value "
            "and phase_func_c_fixed=0.\n"
            f"Stderr: {stderr}"
        )

    return f"convexinv failed with return code {returncode}.\nStderr: {stderr}"


def _validate_minkowski_areas_file(areas_normals_file: str) -> None:
    path = Path(areas_normals_file)
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise AsteroidModelError(f"Malformed minkowski input {path}: file is empty.")

    try:
        num_faces = int(lines[0])
    except ValueError as e:
        raise AsteroidModelError(
            f"Malformed minkowski input {path}: first line must be the number of faces."
        ) from e

    if num_faces <= 0:
        raise AsteroidModelError(
            f"Malformed minkowski input {path}: number of faces must be positive."
        )
    if num_faces > _MINKOWSKI_MAX_FACES:
        raise AsteroidModelError(
            f"Malformed minkowski input {path}: {num_faces} faces exceeds the Fortran limit of {_MINKOWSKI_MAX_FACES}."
        )

    expected_lines = 1 + 2 * num_faces
    if len(lines) != expected_lines:
        raise AsteroidModelError(
            f"Malformed minkowski input {path}: expected {expected_lines} non-empty lines for {num_faces} faces, got {len(lines)}."
        )

    areas = []
    for face_idx in range(num_faces):
        area_line = lines[1 + 2 * face_idx]
        normal_line = lines[2 + 2 * face_idx]
        try:
            area = float(area_line)
            normal = [float(part) for part in normal_line.split()]
        except ValueError as e:
            raise AsteroidModelError(
                f"Malformed minkowski input {path}: invalid numeric value near face {face_idx + 1}."
            ) from e

        if not math.isfinite(area) or area <= 0:
            raise AsteroidModelError(
                f"Malformed minkowski input {path}: face {face_idx + 1} has non-positive or non-finite area."
            )
        areas.append(area)
        if len(normal) != 3:
            raise AsteroidModelError(
                f"Malformed minkowski input {path}: face {face_idx + 1} normal must have exactly 3 values."
            )
        normal_norm = math.sqrt(sum(component * component for component in normal))
        if not math.isfinite(normal_norm) or normal_norm == 0:
            raise AsteroidModelError(
                f"Malformed minkowski input {path}: face {face_idx + 1} has a zero or non-finite normal vector."
            )

    area_ratio = max(areas) / min(areas)
    if area_ratio >= _MINKOWSKI_MAX_AREA_RATIO:
        raise AsteroidModelError(
            f"Malformed minkowski input {path}: numerically degenerate face areas "
            f"(area max/min ratio {area_ratio:.3g}). "
            "Use a better initial spin-axis guess, reduce triangulation rows, or rerun convexinv with different scattering settings."
        )


def _format_minkowski_failure(returncode: int, stderr: str, areas_normals_file: str) -> str:
    if returncode < 0:
        signal_number = -returncode
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f"signal {signal_number}"
        return (
            f"minkowski crashed with signal {signal_name} while reading {areas_normals_file}.\n"
            "The areas/normals file may be invalid or numerically degenerate. "
            "Try a different initial period/spin-axis guess or a lower triangulation row count, "
            "then inspect the generated *_areas.txt file.\n"
            f"Stderr: {stderr}"
        )

    return f"minkowski failed with return code {returncode}.\nStderr: {stderr}"


def run_minkowski(
    areas_normals_file: str,
    pwd_dir: Optional[Union[str, Path]] = None,
    verbose: bool = False,
) -> tuple[np.ndarray, list[list[int]]]:
    """
    Run the minkowski binary to reconstruct a 3D shape from face areas and normals.
    """
    if not MINKOWSKI_EXEC.exists():
        raise FileNotFoundError(
            f"minkowski executable not found at {MINKOWSKI_EXEC}. Please compile 'minkowski.f'."
        )

    _validate_minkowski_areas_file(areas_normals_file)

    run_opts = {}
    if pwd_dir:
        run_opts["cwd"] = str(pwd_dir)

    with open(areas_normals_file, "rb") as f_in:
        try:
            proc = subprocess.Popen(
                [str(MINKOWSKI_EXEC)],
                stdin=f_in,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                **run_opts,
            )
            stdout_data, stderr_data = proc.communicate()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(
                    proc.returncode, [str(MINKOWSKI_EXEC)], stderr=stderr_data
                )
            if verbose and stderr_data.strip():
                print(stderr_data, flush=True)
        except subprocess.CalledProcessError as e:
            raise AsteroidModelError(
                _format_minkowski_failure(
                    e.returncode, e.stderr or "", areas_normals_file
                )
            ) from e

    vertices, faces = _parse_minkowski_output(stdout_data)
    if verbose:
        print(f"  -> Shape: {len(vertices)} vertices, {len(faces)} faces", flush=True)
    return vertices, faces
