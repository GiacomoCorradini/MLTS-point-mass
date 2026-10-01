# %% Setup
import numpy as np
import pandas as pd
import casadi as ca

import mlts_point_mass.define_mesh as mesh
from mlts_point_mass.ggv_constr import GGVConstr
from mlts_point_mass.track import read_track, darboux_to_cartesian


class MLTS:
    """Minimum lap time of a point mass with a G-G-V constraint."""

    X_scale = {"n": 8, "Xi": 0.5, "V": 60, "ax": 15, "ay": 10}
    U_scale = {"ax_ctrl": 15, "ay_ctrl": 10, "ax_dot_ctrl": 15, "ay_dot_ctrl": 10}

    # Default cost weights (lap time, control rate penalty), updated with the
    # ones passed to solution()
    target_weight = {"w__T": 1.0, "w__ax": 1e-5, "w__ay": 1e-5}

    # Default IPOPT options, updated with the ones passed to solution()
    ipopt_default = {
        "ipopt.linear_solver": "mumps",
        "ipopt.print_level": 5,
        "ipopt.max_iter": 1000,
        "ipopt.mu_strategy": "adaptive",
    }

    def __init__(
        self,
        track: str | pd.DataFrame,
        vehicle_data: dict,
        ggv_data: dict,
        ggv_scales: np.ndarray | list | None = None,
    ):
        """Load the track, vehicle and GGV data.

        :param track: track file path or DataFrame (see read_track).
        :param vehicle_data: W_total [m], v_max [m/s]; optional v_min, tau_ax, tau_ay.
        :param ggv_data: GGV samples (V, ax, ay) or FWBW tables.
        :param ggv_scales: grip scales [mu], [mu_ax, mu_ay] or
            [mu_ax_max, mu_ax_min, mu_ay], defaults to 1.
        """
        self.track_data = self._load_racetrack(track)
        self.veh_data = self._get_vehicle_data(vehicle_data)
        self.ggv = GGVConstr(ggv_data, scales=ggv_scales)

    def solution(
        self,
        x0,
        mesh_type="uniform",
        mesh_options=None,
        weights=None,
        ipopt_options=None,
    ):
        """Solve the minimum lap time problem.

        :param x0: constant initial guess [n, Xi, V, ax, ay].
        :param mesh_type: "uniform", "dense_start_end" or "time_uniform".
        :param mesh_options: arguments of the mesh function (see define_mesh),
            defaults to step_size = 1 m for the uniform mesh.
        :param weights: cost weights w__T, w__ax, w__ay, merged with target_weight.
        :param ipopt_options: IPOPT options, merged with ipopt_default.
        :return: solution sampled on the mesh (see README).
        """

        # Cost weights
        weights = {**self.target_weight, **(weights or {})}
        if weights.keys() != self.target_weight.keys():
            raise ValueError(f"weights must be among {list(self.target_weight)}")

        # Define mesh
        s_values, N = self._build_mesh(mesh_type, mesh_options or {})
        print(f"Number of mesh points: {N}")

        # Define the dynamics model
        model = self.dynamics()

        # States and controls
        state_idx = {name: i for i, name in enumerate(model.x_names)}
        control_idx = {name: i for i, name in enumerate(model.u_names)}

        # Initial guess x0 = [n, Xi, V, ax, ay]
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (model.nx,):
            raise ValueError(f"x0 must have {model.nx} elements: {model.x_names}")
        if x0[state_idx["V"]] <= 0:
            raise ValueError("x0: the initial guess of V must be positive")

        # Scale vectors
        x_scale_vec = ca.DM([self.X_scale[name] for name in model.x_names])
        u_scale_vec = ca.DM([self.U_scale[name] for name in model.u_names])
        SX = ca.diag(x_scale_vec)
        SU = ca.diag(u_scale_vec)
        SX_inv = ca.diag(1.0 / x_scale_vec)

        # Initialize the optimization problem
        opti = ca.Opti()
        x_scaled = opti.variable(model.nx, N + 1)
        u_scaled = opti.variable(model.nu, N)
        x_unscaled = SX @ x_scaled
        u_unscaled = SU @ u_scaled

        P = opti.parameter(model.np)
        opti.set_value(P, ca.vertcat(self.veh_data["tau_ax"], self.veh_data["tau_ay"]))

        # Loop over the mesh: states on the nodes s_0..s_N, controls constant on the cells [s_k, s_k+1]
        # fmt: off
        cost = 0
        for k in range(N):
            # ---- Cell k ----
            ds_cell = s_values[k + 1] - s_values[k]
            rho_cell = self.track_data["rho"]((s_values[k] + s_values[k + 1]) / 2)
            x_cell = (x_unscaled[:, k] + x_unscaled[:, k + 1]) / 2

            # Lap time: dt = ds / s_dot (midpoint rule, as the dynamics)
            cost += weights["w__T"] * (ds_cell / model.s_dot(x_cell, P, rho_cell))

            # Dynamics (implicit midpoint)
            opti.subject_to(x_scaled[:, k + 1] - x_scaled[:, k] - ds_cell * SX_inv @ model.f(x_cell, u_unscaled[:, k], P, rho_cell) == 0)

            # ---- Node k+1 ----
            # Controls jump from cell k to the next cell at node k+1. The last
            # node wraps around the lap (node N == node 0, next cell is 0).
            k_next = (k + 1) % N
            ds_node = (ds_cell + s_values[k_next + 1] - s_values[k_next]) / 2
            rho_node = self.track_data["rho"](s_values[k + 1])
            dt_node = ds_node / model.s_dot(x_unscaled[:, k + 1], P, rho_node)

            # Control rate: u_dot = (u_next - u_k) / dt
            ax_ctrl_dot = (u_unscaled[control_idx["ax_ctrl"], k_next] - u_unscaled[control_idx["ax_ctrl"], k]) / dt_node
            ay_ctrl_dot = (u_unscaled[control_idx["ay_ctrl"], k_next] - u_unscaled[control_idx["ay_ctrl"], k]) / dt_node

            # Control rate penalty: W * (u_dot / u_dot_scale)^2 * dt
            cost += weights["w__ax"] * (ax_ctrl_dot / self.U_scale["ax_dot_ctrl"]) ** 2 * dt_node
            cost += weights["w__ay"] * (ay_ctrl_dot / self.U_scale["ay_dot_ctrl"]) ** 2 * dt_node

        # Cyclic condition
        opti.subject_to(x_scaled[:, 0] == x_scaled[:, -1])

        # Path constraints below skip node N (== node 0 by the cyclic condition)

        # Velocity constraints
        opti.subject_to(self.veh_data["v_min"] / self.X_scale["V"] <= x_scaled[state_idx["V"], :-1])
        opti.subject_to(x_scaled[state_idx["V"], :-1] <= self.veh_data["v_max"] / self.X_scale["V"])

        # Path constraints
        opti.subject_to(x_scaled[state_idx["n"], :-1] >= (self.track_data["n_r"](s_values[:-1]).T + self.veh_data["vehHalfWidth"]) / self.X_scale["n"])
        opti.subject_to(x_scaled[state_idx["n"], :-1] <= (self.track_data["n_l"](s_values[:-1]).T - self.veh_data["vehHalfWidth"]) / self.X_scale["n"])

        # GGV constraint: g(ax, ay, V) <= 1
        opti.subject_to(self.ggv.g(x_unscaled[[state_idx["ax"], state_idx["ay"], state_idx["V"]], :-1]) <= 1)
        # fmt: on

        # Minimize cost
        opti.minimize(cost)

        # Initial guess: constant state x0 = [n, Xi, V, ax, ay], controls at steady state (u = a)
        opti.set_initial(x_scaled, ca.repmat(x0 / x_scale_vec, 1, N + 1))
        opti.set_initial(
            u_scaled[control_idx["ax_ctrl"], :],
            x0[state_idx["ax"]] / self.U_scale["ax_ctrl"],
        )
        opti.set_initial(
            u_scaled[control_idx["ay_ctrl"], :],
            x0[state_idx["ay"]] / self.U_scale["ay_ctrl"],
        )

        opti.solver("ipopt", {**self.ipopt_default, **(ipopt_options or {})})

        try:
            sol = opti.solve()
        except RuntimeError:
            sol = opti.debug

        # Solver outcome
        stats = opti.stats()
        solver_info = {
            "success": bool(stats["success"]),
            "status": stats["return_status"],
            "iterations": int(stats["iter_count"]),
            "solve_time": float(stats.get("t_wall_total", np.nan)),
        }
        if not solver_info["success"]:
            print(
                f"WARNING: solver did not converge ({solver_info['status']}), returning the last iterate."
            )

        # Extract solution
        x_sol = np.array(SX @ sol.value(x_scaled))

        # Compute time (midpoint rule on the cells, as the cost)
        p_values = ca.vertcat(self.veh_data["tau_ax"], self.veh_data["tau_ay"])
        x_cell = (x_sol[:, :-1] + x_sol[:, 1:]) / 2
        rho_cell = self.track_data["rho"]((s_values[:-1] + s_values[1:]) / 2).T
        s_dot_cell = np.array(model.s_dot(x_cell, p_values, rho_cell)).squeeze()
        t_values = np.concatenate(([0.0], np.cumsum(np.diff(s_values) / s_dot_cell)))

        # Road info
        road = lambda key: np.array(self.track_data[key](s_values)).squeeze()
        x_mid_line, y_mid_line, z_mid_line = road("x"), road("y"), road("z")

        # Trajectory in cartesian coordinates
        x_trj, y_trj, z_trj = darboux_to_cartesian(
            x_mid_line,
            y_mid_line,
            z_mid_line,
            road("theta"),
            road("banking"),
            road("slope"),
            x_sol[state_idx["n"], :],
        )

        # Travelled distance along the trajectory
        ds_trj = np.sqrt(
            np.diff(x_trj) ** 2 + np.diff(y_trj) ** 2 + np.diff(z_trj) ** 2
        )
        distance = np.concatenate(([0.0], np.cumsum(ds_trj)))

        # Yaw rate: psi_dot = (Xi' + kappa) * s_dot = ay / V
        yaw_rate = x_sol[state_idx["ay"], :] / x_sol[state_idx["V"], :]

        # Trajectory curvature: kappa_trj = psi_dot / V = ay / V^2
        kappa_trj = yaw_rate / x_sol[state_idx["V"], :]

        print(f"Total time to complete the track: {t_values[-1]:.2f} seconds")

        return {
            **solver_info,
            "abscissa": s_values,
            "distance": distance,
            "time": t_values,
            "n": x_sol[state_idx["n"], :],
            "kappa": road("rho"),
            "kappa_trj": kappa_trj,
            "Xi": x_sol[state_idx["Xi"], :],
            "yaw_rate": yaw_rate,
            "V": x_sol[state_idx["V"], :],
            "ax": x_sol[state_idx["ax"], :],
            "ay": x_sol[state_idx["ay"], :],
            "x_trj": np.array(x_trj).squeeze(),
            "y_trj": np.array(y_trj).squeeze(),
            "z_trj": np.array(z_trj).squeeze(),
            "x_mid_line": x_mid_line,
            "y_mid_line": y_mid_line,
            "z_mid_line": z_mid_line,
            "x_margin_L": road("x_margin_L"),
            "y_margin_L": road("y_margin_L"),
            "z_margin_L": road("z_margin_L"),
            "x_margin_R": road("x_margin_R"),
            "y_margin_R": road("y_margin_R"),
            "z_margin_R": road("z_margin_R"),
        }

    #   ____       _            _
    #  |  _ \ _ __(_)_   ____ _| |_ ___
    #  | |_) | '__| \ \ / / _` | __/ _ \
    #  |  __/| |  | |\ V / (_| | ||  __/
    #  |_|   |_|  |_| \_/ \__,_|\__\___|

    def dynamics(self):
        """Point mass dynamics in curvilinear coordinates, derivatives w.r.t. s."""
        model = ca.types.SimpleNamespace()

        # Curvature
        kappa = ca.SX.sym("kappa")

        # State variables
        n = ca.SX.sym("n")
        Xi = ca.SX.sym("Xi")
        V = ca.SX.sym("V")
        ax = ca.SX.sym("ax")
        ay = ca.SX.sym("ay")
        model.x = ca.vertcat(n, Xi, V, ax, ay)
        model.x_names = [model.x[i].name() for i in range(model.x.size1())]

        # Control inputs
        ax_ctrl = ca.SX.sym("ax_ctrl")
        ay_ctrl = ca.SX.sym("ay_ctrl")
        model.u = ca.vertcat(ax_ctrl, ay_ctrl)
        model.u_names = [model.u[i].name() for i in range(model.u.size1())]

        # Parameters
        tau_ax = ca.SX.sym("tau_ax")
        tau_ay = ca.SX.sym("tau_ay")
        model.p = ca.vertcat(tau_ax, tau_ay)
        model.p_names = [model.p[i].name() for i in range(model.p.size1())]

        # S_dot function
        s_dot = (V * ca.cos(Xi)) / (1 - kappa * n)
        model.s_dot = ca.Function("s_dot", [model.x, model.p, kappa], [s_dot])

        # Dynamics equations (derivatives w.r.t. s)
        n_dot = ca.sin(Xi) * V / s_dot
        Xi_dot = ay / V / s_dot - kappa
        V_dot = ax / s_dot
        ax_dot = (ax_ctrl - ax) / tau_ax / s_dot
        ay_dot = (ay_ctrl - ay) / tau_ay / s_dot
        model.x_dot = ca.vertcat(n_dot, Xi_dot, V_dot, ax_dot, ay_dot)

        model.f = ca.Function(
            "f",
            [model.x, model.u, model.p, kappa],
            [model.x_dot],
        )

        model.nx = model.x.size1()
        model.nu = model.u.size1()
        model.np = model.p.size1()

        return model

    def _build_mesh(self, mesh_type: str, mesh_options: dict) -> tuple[np.ndarray, int]:
        """Mesh nodes along s and number of cells."""
        s = self.track_data["s_values"]
        if mesh_type == "uniform":
            return mesh.build_uniform_spatial_mesh(
                s, **{"step_size": 1.0, **mesh_options}
            )
        if mesh_type == "dense_start_end":
            return mesh.build_dense_start_end_mesh(s, **mesh_options)
        if mesh_type == "time_uniform":
            kappa = np.array(self.track_data["rho"](s)).squeeze()
            return mesh.build_time_uniform_mesh(s, kappa, mesh_options)
        raise ValueError(
            'mesh_type must be "uniform", "dense_start_end" or "time_uniform"'
        )

    def _load_racetrack(self, track: str | pd.DataFrame) -> dict:
        """Splines of the track data and borders along s."""
        track = read_track(track)
        abscissa = track["abscissa"].to_numpy()

        # Track borders (without kerbs)
        mid_line = [
            track[col].to_numpy()
            for col in ("x_mid_line", "y_mid_line", "elevation", "dir_mid_line")
        ]
        bank, slope = track["banking"].to_numpy(), track["slope"].to_numpy()
        width_L = track["width_no_kerbs_L"].to_numpy()
        width_R = track["width_no_kerbs_R"].to_numpy()
        x_L, y_L, z_L = darboux_to_cartesian(*mid_line, bank, slope, width_L)
        x_R, y_R, z_R = darboux_to_cartesian(*mid_line, bank, slope, -width_R)

        # Create the splines
        spline = lambda name, values: ca.interpolant(
            name, "bspline", [abscissa], values
        )
        return {
            "s_values": abscissa,
            "x": spline("x", mid_line[0]),
            "y": spline("y", mid_line[1]),
            "z": spline("z", mid_line[2]),
            "theta": spline("theta", mid_line[3]),
            "banking": spline("banking", bank),
            "slope": spline("slope", slope),
            "rho": spline("rho", track["curvature"].to_numpy()),
            "n_l": spline("n_l", width_L),
            "n_r": spline("n_r", -width_R),
            "x_margin_L": spline("x_margin_L", x_L),
            "y_margin_L": spline("y_margin_L", y_L),
            "z_margin_L": spline("z_margin_L", z_L),
            "x_margin_R": spline("x_margin_R", x_R),
            "y_margin_R": spline("y_margin_R", y_R),
            "z_margin_R": spline("z_margin_R", z_R),
        }

    def _get_vehicle_data(self, data: dict) -> dict:
        """Check the vehicle data and fill the defaults."""
        width = data.get("W_total", data.get("vehWidth"))
        if width is None:
            raise KeyError(
                "vehicle_data must contain the vehicle width as 'W_total' (or 'vehWidth')"
            )
        if "v_max" not in data:
            raise KeyError("vehicle_data must contain the maximum speed 'v_max'")

        veh_data = {
            "vehHalfWidth": width / 2,
            "tau_ax": data.get("tau_ax", 0.03),
            "tau_ay": data.get("tau_ay", 0.03),
            "v_min": data.get("v_min", 5.0),
            "v_max": data["v_max"],
        }
        if veh_data["v_min"] >= veh_data["v_max"]:
            raise ValueError("v_min must be lower than v_max")
        return veh_data
