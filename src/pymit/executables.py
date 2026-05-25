import math
import re
import signal
import subprocess
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .errors import AsteroidModelError
from .paths import CONVEXINV_EXEC, MINKOWSKI_EXEC, PERIOD_SCAN_EXEC
from .shape import _parse_minkowski_output


_MINKOWSKI_MAX_FACES = 6000
_MINKOWSKI_MAX_AREA_RATIO = 1_000_000.0
_MINKOWSKI_TIMEOUT_SECONDS = 300.0


def run_convexinv(
    param_file: str,
    lightcurve_file: str,
    output_areas_file: str,
    output_lc_file: str,
    verbose: bool = False,
    stdout_log_file: Optional[Union[str, Path]] = None,
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
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                collected_stdout.append(line)
                if verbose and not line.startswith("f<0"):
                    print(line, end="", flush=True)
            proc.wait()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(
                    proc.returncode, cmd, stderr="".join(collected_stdout)
                )
        except subprocess.CalledProcessError as e:
            raise AsteroidModelError(
                _format_convexinv_failure(
                    e.returncode, e.stderr or "", param_file
                )
            ) from e

    result_stdout = "".join(collected_stdout)
    if stdout_log_file is not None:
        Path(stdout_log_file).write_text(result_stdout)

    return _parse_convexinv_output(result_stdout)


def run_period_scan(
    param_file: str,
    lightcurve_file: str,
    output_periods_file: str,
    verbose: bool = False,
    stdout_log_file: Optional[Union[str, Path]] = None,
) -> None:
    """
    Run the DAMIT period_scan binary to estimate a period before convexinv.
    """
    if not PERIOD_SCAN_EXEC.exists():
        raise FileNotFoundError(
            f"period_scan executable not found at {PERIOD_SCAN_EXEC}. Please run 'make' in {PERIOD_SCAN_EXEC.parent}."
        )

    cmd = [str(PERIOD_SCAN_EXEC)]
    if verbose:
        cmd.append("-v")
    cmd.extend([param_file, output_periods_file])

    collected_stdout = []
    with open(lightcurve_file, "rb") as f_in:
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=f_in,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                collected_stdout.append(line)
                if verbose:
                    print(line, end="", flush=True)
            proc.wait()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(
                    proc.returncode, cmd, stderr="".join(collected_stdout)
                )
        except subprocess.CalledProcessError as e:
            raise AsteroidModelError(
                f"period_scan failed with return code {e.returncode}.\nStderr: {e.stderr or ''}"
            ) from e

    if stdout_log_file is not None:
        Path(stdout_log_file).write_text("".join(collected_stdout))


def _parse_convexinv_output(result_stdout: str) -> dict:
    fit = {}
    chi_re = re.compile(
        r"\bchi2\s+([-+0-9.eE]+)\s+dev\s+([-+0-9.eE]+)", re.IGNORECASE
    )
    dark_re = re.compile(
        r"dark facet with area\s+([-+0-9.eE]+)\s*%", re.IGNORECASE
    )

    for line in result_stdout.splitlines():
        chi_match = chi_re.search(line)
        if chi_match:
            fit["chi_square"] = float(chi_match.group(1))
            fit["dev"] = float(chi_match.group(2))

        dark_match = dark_re.search(line)
        if dark_match:
            fit["shadow_percent"] = float(dark_match.group(1))

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


def _convexinv_parameter_label(line_index: int, comment: str) -> str:
    normalized = comment.lower()
    if "lambda" in normalized:
        return "initial lambda"
    if "beta" in normalized:
        return "initial beta"
    if "period" in normalized:
        return "initial period"
    if "param. 'a'" in normalized:
        return "phase function a / DAMIT p2"
    if "param. 'd'" in normalized:
        return "phase function d / DAMIT p3"
    if "param. 'k'" in normalized:
        return "phase function k / DAMIT p4"
    if "lambert" in normalized or "coefficient 'c'" in normalized:
        return "Lambert coefficient c / DAMIT p1"
    return f"parameter on line {line_index}"


def _summarize_convexinv_free_parameters(param_file: str) -> tuple[list[str], bool | None]:
    free_parameters = []
    lambert_is_fixed = None

    try:
        lines = Path(param_file).read_text().splitlines()
    except OSError:
        return free_parameters, lambert_is_fixed

    for line_index, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) < 2:
            continue

        flag = parts[1]
        if flag not in {"0", "1"}:
            continue

        comment = " ".join(parts[2:])
        label = _convexinv_parameter_label(line_index, comment)
        is_free = flag == "1"
        if label == "Lambert coefficient c / DAMIT p1":
            lambert_is_fixed = not is_free
        if is_free:
            free_parameters.append(label)

    return free_parameters, lambert_is_fixed


def _format_convexinv_failure(returncode: int, stderr: str, param_file: str) -> str:
    if "Singular Matrix" in stderr:
        free_parameters, lambert_is_fixed = _summarize_convexinv_free_parameters(param_file)
        if free_parameters:
            free_summary = f"Free parameters in {param_file}: {', '.join(free_parameters)}."
        else:
            free_summary = (
                f"No free 0/1 parameters could be identified in {param_file}."
            )

        if lambert_is_fixed is True:
            lambert_summary = "Lambert coefficient c is fixed."
        elif lambert_is_fixed is False:
            lambert_summary = (
                "Lambert coefficient c is free; try fixing DAMIT p1/c first."
            )
        else:
            lambert_summary = "Lambert coefficient c status could not be determined."

        return (
            f"convexinv failed because its linear solve became singular while using {param_file}.\n"
            "This often happens when too many parameters are free or poorly constrained for the available lightcurve data.\n"
            f"{free_summary}\n"
            f"{lambert_summary}\n"
            "Try fixing some free parameters, especially phase-function parameters p2/p3/p4, "
            "then rerun convexinv.\n"
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
    timeout_seconds: Optional[float] = _MINKOWSKI_TIMEOUT_SECONDS,
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
            stdout_data, stderr_data = proc.communicate(timeout=timeout_seconds)
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(
                    proc.returncode, [str(MINKOWSKI_EXEC)], stderr=stderr_data
                )
            if verbose and stderr_data.strip():
                print(stderr_data, flush=True)
        except subprocess.TimeoutExpired as e:
            proc.kill()
            proc.communicate()
            raise AsteroidModelError(
                f"minkowski timed out after {timeout_seconds:g} seconds while reading {areas_normals_file}.\n"
                "The areas/normals file may be numerically degenerate and can make the Fortran reconstruction stall. "
                "Try pole grid scan, a different initial spin-axis/period guess, or a lower triangulation row count."
            ) from e
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
