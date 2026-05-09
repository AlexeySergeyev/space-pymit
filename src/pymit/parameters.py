def create_convexinv_param_file(options: dict, output_file: str) -> None:
    """
    Programmatically create the convexinv parameter file.

    Default options mirror the standard `input_convexinv` typically used.
    """
    defaults = {
        "initial_lambda": 220,
        "initial_lambda_fixed": 1,
        "initial_beta": 0,
        "initial_beta_fixed": 1,
        "initial_period": 5.76198,
        "initial_period_fixed": 1,
        "zero_time": 0,
        "initial_rotation_angle": 0,
        "convexity_regularization": 0.1,
        "spherical_harmonics_degree": 6,
        "spherical_harmonics_order": 6,
        "number_of_rows": 8,
        "phase_func_a": 0.5,
        "phase_func_a_fixed": 0,
        "phase_func_d": 0.1,
        "phase_func_d_fixed": 0,
        "phase_func_k": -1.05,
        "phase_func_k_fixed": 0,
        "phase_func_c": 0.1,
        "phase_func_c_fixed": 0,
        "iteration_stop_condition": 50,
    }

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
        f"{opts['phase_func_c']}\t{opts['phase_func_c_fixed']}\tLambert coefficient 'c' (0/1 - fixed/free)",
        f"{opts['iteration_stop_condition']}\t\titeration stop condition",
    ]

    with open(output_file, "w") as f:
        f.write("\n".join(lines) + "\n")


def create_conjgradinv_param_file(options: dict, output_file: str) -> None:
    """Programmatically create the conjgradinv parameter file used by Minkowski."""
    defaults = {
        "convexity_weight": 0.2,
        "number_of_rows": 8,
        "number_of_iterations": 100,
    }

    opts = {**defaults, **(options or {})}

    lines = [
        f"{opts['convexity_weight']}\t\t\tconvexity weight",
        f"{opts['number_of_rows']}\t\t\tnumber of rows",
        f"{opts['number_of_iterations']}\t\t\tnumber of iterations",
    ]

    with open(output_file, "w") as f:
        f.write("\n".join(lines) + "\n")
