import numpy as np


def build_uniform_spatial_mesh(
    abscissa: np.ndarray | list,
    step_size: float,
) -> tuple[np.ndarray, int]:
    """Uniform mesh with cells of about step_size.

    :param abscissa: track abscissa in m.
    :param step_size: cell length in m.
    :return: mesh nodes and number of cells.
    """

    if step_size <= 0:
        raise ValueError("step_size must be positive")

    s_start = float(abscissa[0])
    s_end = float(abscissa[-1])
    track_length = s_end - s_start

    num_nodes = int(np.ceil(track_length / step_size)) + 1
    nodes = np.linspace(s_start, s_end, num_nodes)

    # monotonicity check
    if np.any(np.diff(nodes) <= 0):
        raise ValueError("Generated mesh is not strictly increasing")

    return nodes, nodes.size - 1


def build_dense_start_end_mesh(
    abscissa: np.ndarray | list,
    coarse_step: float,
    dense_step: float,
    dense_length: float = 50.0,
) -> tuple[np.ndarray, int]:
    """Mesh dense close to the start and the end of the track, coarse elsewhere.

    :param abscissa: track abscissa in m.
    :param coarse_step: cell length far from start and end in m.
    :param dense_step: cell length close to start and end in m.
    :param dense_length: length of the dense regions in m, defaults to 50.
    :return: mesh nodes and number of cells.
    """

    if dense_step <= 0 or coarse_step <= 0:
        raise ValueError("dense_step and coarse_step must be positive")

    s_start = float(abscissa[0])
    s_end = float(abscissa[-1])

    s_current = s_start
    nodes = [s_current]
    step = dense_step

    while True:
        # dense mesh near start and end, coarse elsewhere (distances measured
        # from s_start / s_end, so the track may not start at s = 0)
        if s_current - s_start < dense_length or s_end - s_current < dense_length:
            step = dense_step
        else:
            step = coarse_step

        s_current += step
        if s_current >= s_end:
            break
        nodes.append(s_current)

    # Close the mesh exactly on s_end. Merge a too short last cell into the
    # previous one, so no cell is shorter than half its step.
    if len(nodes) > 1 and s_end - nodes[-1] < 0.5 * step:
        nodes.pop()
    nodes.append(s_end)
    nodes = np.array(nodes)

    # monotonicity check
    if np.any(np.diff(nodes) <= 0):
        raise ValueError("Generated mesh is not strictly increasing")

    return nodes, nodes.size - 1


def build_time_uniform_mesh(
    abscissa: np.ndarray | list, curvature: np.ndarray | list, mesh_cfg: dict
) -> tuple[np.ndarray, int]:
    """Mesh with cells of about dt seconds along a reference speed profile.

    :param abscissa: track abscissa in m.
    :param curvature: track curvature in 1/m.
    :param mesh_cfg: dt, hmax, maxacc, minacc, curvature_velocity; optional vmin,
        vmax (see examples/plot_mesh.py).
    :return: mesh nodes and number of cells.
    """

    abscissa = np.asarray(abscissa, dtype=float)
    curvature = np.asarray(curvature, dtype=float)

    # --- mesh parameters ---
    dt = mesh_cfg["dt"]
    h_max = mesh_cfg["hmax"]
    # Accelerations as magnitudes: minacc may be given as a (negative) deceleration
    acc_max = abs(mesh_cfg["maxacc"])
    acc_min = abs(mesh_cfg["minacc"])

    if dt <= 0 or h_max <= 0:
        raise ValueError("dt and hmax must be positive")

    # Optional velocity saturation: vmin bounds the smallest cell (vmin * dt),
    # vmax (together with hmax) the largest one
    v_sat_min = mesh_cfg.get("vmin", 0.0)
    v_sat_max = min(mesh_cfg.get("vmax", np.inf), h_max / dt)
    if v_sat_min < 0 or v_sat_min >= v_sat_max:
        raise ValueError(
            "vmin must be non-negative and lower than min(vmax, hmax / dt)"
        )

    curvature_velocity_map = sorted(
        mesh_cfg["curvature_velocity"], key=lambda p: p["k"]
    )

    # --- curvature -> velocity map ---
    # Linear interpolation, held constant outside the knots: a cubic spline
    # overshoots and extrapolates (possibly to v <= 0) beyond the last knot
    curvature_knots = np.array([p["k"] for p in curvature_velocity_map], dtype=float)
    velocity_knots = np.array([p["v"] for p in curvature_velocity_map], dtype=float)

    # --- curvature-limited velocity profile ---
    v_limit = np.interp(np.abs(curvature), curvature_knots, velocity_knots)
    v_limit = np.clip(v_limit, v_sat_min, v_sat_max)
    if np.any(v_limit <= 0):
        raise ValueError(
            "curvature_velocity must map to positive velocities (or set vmin > 0)"
        )

    # --- forward pass: acceleration limit (constant acceleration over ds) ---
    for i in range(1, len(abscissa)):
        ds = abscissa[i] - abscissa[i - 1]
        v_allowed = np.sqrt(v_limit[i - 1] ** 2 + 2 * acc_max * ds)
        v_limit[i] = min(v_limit[i], v_allowed)

    # --- backward pass: braking limit ---
    for i in reversed(range(len(abscissa) - 1)):
        ds = abscissa[i + 1] - abscissa[i]
        v_allowed = np.sqrt(v_limit[i + 1] ** 2 + 2 * acc_min * ds)
        v_limit[i] = min(v_limit[i], v_allowed)

    # --- build cumulative time index ---
    ds_target = np.minimum(v_limit[:-1], v_limit[1:]) * dt
    time_index = np.concatenate(([0.0], np.cumsum(np.diff(abscissa) / ds_target)))
    cumulative_steps = time_index[-1]

    num_intervals = int(np.ceil(cumulative_steps))
    if num_intervals <= 0:
        nodes = np.array([abscissa[0], abscissa[-1]])
        return nodes, nodes.size - 1

    # --- generate final mesh ---
    # Inverse mapping time index -> arc length. Linear, so it stays monotone
    # (a cubic spline can overshoot and break the mesh ordering)
    step = cumulative_steps / num_intervals
    nodes = np.interp(np.arange(num_intervals + 1) * step, time_index, abscissa)
    nodes[0], nodes[-1] = abscissa[0], abscissa[-1]

    # --- monotonicity check ---
    if np.any(np.diff(nodes) <= 0):
        raise ValueError("Generated mesh is not strictly increasing")

    return nodes, nodes.size - 1
