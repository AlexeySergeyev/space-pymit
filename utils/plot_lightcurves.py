import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _default_paths():
    # Adjust paths based on whether run from project root or utils dir
    base_dir = Path(__file__).parent.parent
    return {
        "input_file": base_dir / "damit" / "test_lcs_abs",
        "model_output_file": base_dir / "pipeline_output" / "test_asteroid_lcs.txt",
        "output_dir": base_dir / "assets",
    }


def load_model_period_hours(best_json):
    """Read the fitted model rotation period from a pole-scan best JSON file."""
    best_path = Path(best_json)
    data = json.loads(best_path.read_text())
    period = data.get("fitted_period")
    if period is None:
        period = data.get("fit_result", {}).get("period")
    if period is None:
        raise ValueError(
            f"No fitted_period or fit_result.period found in {best_path}"
        )

    period_hours = float(period)
    if period_hours == 0:
        raise ValueError(f"Model rotation period must be non-zero in {best_path}")
    return period_hours


def _candidate_param_paths(best_json, param_file):
    param_path = Path(param_file)
    if param_path.is_absolute():
        return [param_path]

    best_path = Path(best_json)
    return [
        param_path,
        best_path.parent / param_path,
        best_path.parent / param_path.name,
    ]


def _parse_zero_time_from_param_file(param_file):
    lines = [
        line.strip()
        for line in Path(param_file).read_text().splitlines()
        if line.strip()
    ]
    for line in lines:
        if "zero time" in line.lower():
            return float(line.split()[0])

    if len(lines) >= 4:
        return float(lines[3].split()[0])
    raise ValueError(f"Could not parse zero time from {param_file}")


def first_observation_jd(input_file):
    """Return the first observation JD from a native DAMIT lightcurve file."""
    with open(input_file, "r") as f_in:
        f_in.readline()
        f_in.readline()
        row = f_in.readline().split()
    if not row:
        raise ValueError(f"No observations found in {input_file}")
    return float(row[0])


def load_model_zero_time(best_json, input_file):
    """Read zero time from the model parameter file, or use first observation JD."""
    best_path = Path(best_json)
    data = json.loads(best_path.read_text())
    param_file = data.get("param_file")
    if param_file:
        for candidate in _candidate_param_paths(best_path, param_file):
            if candidate.exists():
                return _parse_zero_time_from_param_file(candidate)
        print(
            f"Could not find parameter file {param_file}; "
            "using first observation JD for folded phase."
        )
    return first_observation_jd(input_file)


def fold_rotation_phase(jd, zero_time, period_hours):
    period_days = period_hours / 24.0
    if period_days == 0:
        raise ValueError("period_hours must be non-zero.")
    return ((float(jd) - zero_time) / period_days) % 1.0


def find_best_json(model_output_file):
    """Find a likely pole-scan best JSON for a model lightcurve output file."""
    model_path = Path(model_output_file)
    stem = model_path.stem
    parents = [model_path.parent, *model_path.parents[1:4]]

    candidates = []
    if stem.endswith("_lc_output"):
        prefix = stem[: -len("_lc_output")]
        candidates.extend(
            parent / f"{prefix}_pole_scan_best.json" for parent in parents
        )
    candidates.extend(parent / f"{stem}_pole_scan_best.json" for parent in parents)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    for parent in parents:
        matches = sorted(parent.glob("*_pole_scan_best.json"))
        if len(matches) == 1:
            return matches[0]
    return None


def read_lightcurve_records(input_file, model_output_file):
    """Read aligned observed and modeled lightcurve rows."""
    with open(model_output_file, "r") as f_out:
        model_values = [
            float(line.strip())
            for line in f_out
            if line.strip()
        ]

    records = []
    model_index = 0
    with open(input_file, "r") as f_in:
        n_curves = int(f_in.readline().strip())
        for curve_index in range(1, n_curves + 1):
            header = f_in.readline().split()
            if not header:
                raise ValueError(f"Missing lightcurve header for curve {curve_index}")
            n_pts = int(header[0])
            t0 = None

            for _ in range(n_pts):
                parts = f_in.readline().split()
                if len(parts) < 2:
                    raise ValueError(
                        f"Missing observation row in curve {curve_index}"
                    )
                if model_index >= len(model_values):
                    raise ValueError(
                        "Model output has fewer rows than the observation file: "
                        f"{len(model_values)} model values for at least "
                        f"{model_index + 1} observations."
                    )

                jd = float(parts[0])
                if t0 is None:
                    t0 = jd
                records.append(
                    {
                        "curve": curve_index,
                        "time": jd - t0,
                        "jd": jd,
                        "observed": float(parts[1]),
                        "model": model_values[model_index],
                    }
                )
                model_index += 1

    if model_index != len(model_values):
        raise ValueError(
            "Model output row count does not match observations: "
            f"{len(model_values)} model values for {model_index} observations."
        )
    return records


def _records_for_display(records, max_curves):
    return [record for record in records if record["curve"] <= max_curves]


def plot_time_lightcurves(records, save_path, max_curves=3):
    """Save the observed-vs-modeled lightcurve plot against relative time."""
    plt.figure(figsize=(10, 6))

    for curve_index in range(1, max_curves + 1):
        curve_records = [
            record for record in records if record["curve"] == curve_index
        ]
        if not curve_records:
            continue

        x = [record["time"] for record in curve_records]
        y_obs = [record["observed"] for record in curve_records]
        y_mod = [record["model"] for record in curve_records]
        plt.plot(x, y_obs, "o", label=f"Observed Curve {curve_index}")
        plt.plot(x, y_mod, "-", label=f"Modeled Curve {curve_index}")

    plt.xlabel("Time (Days from curve start)")
    plt.ylabel("Brightness")
    plt.title("Observed vs Modeled Light Curves (Sample)")
    plt.legend()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_folded_residuals(
    records,
    period_hours,
    zero_time,
    save_path,
    max_curves=3,
):
    """Save folded observed/model brightness and residuals vs rotation phase."""
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X"]
    fig, (ax_lc, ax_resid) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    for curve_index in range(1, max_curves + 1):
        curve_records = [
            record for record in records if record["curve"] == curve_index
        ]
        if not curve_records:
            continue

        phases = [
            fold_rotation_phase(record["jd"], zero_time, period_hours)
            for record in curve_records
        ]
        observed = [record["observed"] for record in curve_records]
        modeled = [record["model"] for record in curve_records]
        residuals = [
            record["observed"] - record["model"] for record in curve_records
        ]
        order = sorted(range(len(phases)), key=phases.__getitem__)
        phases_sorted = [phases[index] for index in order]
        modeled_sorted = [modeled[index] for index in order]

        ax_lc.plot(
            phases,
            observed,
            markers[curve_index % len(markers)],
            markersize=4,
            color="C{}".format(curve_index % 10),
            label=f"Observed Curve {curve_index}",
        )
        ax_lc.plot(
            phases_sorted,
            modeled_sorted,
            "-",
            linewidth=1.3,
            color="C{}".format(curve_index % 10),
            label=f"Modeled Curve {curve_index}",
        )
        ax_resid.plot(
            phases,
            residuals,
            markers[curve_index % len(markers)],
            markersize=4,
            color="C{}".format(curve_index % 10),
            label=f"Curve {curve_index}",
        )

    ax_lc.set_ylabel("Brightness")
    ax_lc.set_title(
        f"Observed and Modeled Light Curves Folded by Period ({period_hours:.6g} h)"
    )
    ax_lc.legend(fontsize="small", ncols=2, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax_resid.axhline(0.0, color="0.3", linewidth=1.0)
    ax_resid.set_xlabel("Rotation Phase")
    ax_resid.set_ylabel("Observed - Model")
    ax_resid.set_xlim(0.0, 1.0)
    ax_resid.legend(fontsize="small", ncols=2, bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def resolve_best_json(best_json, model_output_file):
    if best_json:
        best_path = Path(best_json)
        if best_path.exists():
            return best_path
        print(
            f"Best JSON {best_path} was not found; "
            "skipping folded residual plot."
        )
        return None
    return find_best_json(model_output_file)


def plot_lcs_offline(
    input_file=None,
    model_output_file=None,
    best_json=None,
    output_dir=None,
    max_curves=3,
):
    defaults = _default_paths()
    input_file = Path(input_file) if input_file else defaults["input_file"]
    output_file = (
        Path(model_output_file)
        if model_output_file
        else defaults["model_output_file"]
    )
    assets_dir = Path(output_dir) if output_dir else defaults["output_dir"]

    assets_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists() or not output_file.exists():
        print(f"Required files not found. Ensure pipeline has been run first.")
        return

    records = read_lightcurve_records(input_file, output_file)
    save_path = assets_dir / "lightcurves.png"
    plot_time_lightcurves(records, save_path, max_curves=max_curves)
    print(f"Saved plot to {save_path}")

    best_path = resolve_best_json(best_json, output_file)
    if best_path is None:
        print(
            "Folded residual plot skipped: provide --best-json or place a "
            "*_pole_scan_best.json file near the model output."
        )
        return

    period_hours = load_model_period_hours(best_path)
    zero_time = load_model_zero_time(best_path, input_file)
    folded_path = assets_dir / "lightcurves_folded_residuals.png"
    plot_folded_residuals(
        _records_for_display(records, max_curves),
        period_hours,
        zero_time,
        folded_path,
        max_curves=max_curves,
    )
    print(f"Saved folded residual plot to {folded_path}")


def _parse_args():
    defaults = _default_paths()
    parser = argparse.ArgumentParser(
        description="Plot observed, modeled, and folded residual lightcurves."
    )
    parser.add_argument(
        "--input-file",
        default=str(defaults["input_file"]),
        help="Native DAMIT/convexinv observation lightcurve file.",
    )
    parser.add_argument(
        "--model-output-file",
        default=str(defaults["model_output_file"]),
        help="convexinv modeled lightcurve output file.",
    )
    parser.add_argument(
        "--best-json",
        default=None,
        help="Pole-scan best JSON containing the fitted model period.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(defaults["output_dir"]),
        help="Directory where PNG plots are written.",
    )
    parser.add_argument(
        "--max-curves",
        type=int,
        default=3,
        help="Maximum number of lightcurve blocks to plot.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    plot_lcs_offline(
        input_file=args.input_file,
        model_output_file=args.model_output_file,
        best_json=args.best_json,
        output_dir=args.output_dir,
        max_curves=args.max_curves,
    )
