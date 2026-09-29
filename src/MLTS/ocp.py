# %% Setup
import numpy as np
import casadi as ca

from pytelemsys.pytrack import TrackData
from pytelemsys.utils.conversion import darboux_to_cartesian

import MLTS.define_mesh as mesh
from MLTS.ggv_constr import GGVConstr


class MLTS:

    X_scale = {"n": 8, "Xi": 0.5, "V": 60, "ax": 15, "ay": 10}
    U_scale = {"ax_ctrl": 15, "ay_ctrl": 10, "ax_dot_ctrl": 600, "ay_dot_ctrl": 100}

    # Weights of the control rate penalty
    target_weight = {"w__T": 1.0, "w__ax": 4.5e-3, "w__ay": 5e-2}

    # Default IPOPT options, updated with the ones passed to solution()
    ipopt_default = {
        "ipopt.linear_solver": "mumps",
        "ipopt.print_level": 5,
        "ipopt.max_iter": 1000,
        "ipopt.mu_strategy": "adaptive",
    }

    def __init__(
        self,
        track: TrackData,
        vehicle_data: dict,
        ggv_data: dict,
        ggv_scales: np.ndarray | list | None = None,
    ):
        self.track_data = self._load_racetrack(track)
        self.veh_data = self._get_vehicle_data(vehicle_data)
        self.ggv = GGVConstr(ggv_data, scales=ggv_scales)

    def solution(self, x0, step_size=1.0, ipopt_options=None):

        # Define mesh
        s_values, N = mesh.build_uniform_spatial_mesh(
            self.track_data["s_values"], step_size
        )
        print(f"Number of mesh points: {N}")

        # Define the dynamics model
        model = self.dynamics()

        # States and controls
        state_idx = {name: i for i, name in enumerate(model.x_names)}
        control_idx = {name: i for i, name in enumerate(model.u_names)}

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
            cost += self.target_weight["w__T"] * (ds_cell / model.s_dot(x_cell, P, rho_cell))

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
            cost += self.target_weight["w__ax"] * (ax_ctrl_dot / self.U_scale["ax_dot_ctrl"]) ** 2 * dt_node
            cost += self.target_weight["w__ay"] * (ay_ctrl_dot / self.U_scale["ay_dot_ctrl"]) ** 2 * dt_node

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
        x0 = np.asarray(x0, dtype=float)
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
        x_M, y_M, z_M = road("x"), road("y"), road("z")

        # Trajectory in cartesian coordinates
        x_trj, y_trj, z_trj = darboux_to_cartesian(
            x_M,
            y_M,
            z_M,
            road("theta"),
            road("banking"),
            road("slope"),
            x_sol[state_idx["n"], :],
        )

        print(f"Total time to complete the track: {t_values[-1]:.2f} seconds")

        return {
            **solver_info,
            "s": s_values,
            "time": t_values,
            "n": x_sol[state_idx["n"], :],
            "kappa": road("rho"),
            "Xi": x_sol[state_idx["Xi"], :],
            "V": x_sol[state_idx["V"], :],
            "ax": x_sol[state_idx["ax"], :],
            "ay": x_sol[state_idx["ay"], :],
            "x_trj": np.array(x_trj).squeeze(),
            "y_trj": np.array(y_trj).squeeze(),
            "z_trj": np.array(z_trj).squeeze(),
            "x_M": x_M,
            "y_M": y_M,
            "z_M": z_M,
            "x_L": road("x_L"),
            "y_L": road("y_L"),
            "z_L": road("z_L"),
            "x_R": road("x_R"),
            "y_R": road("y_R"),
            "z_R": road("z_R"),
        }

    #   ____       _            _
    #  |  _ \ _ __(_)_   ____ _| |_ ___
    #  | |_) | '__| \ \ / / _` | __/ _ \
    #  |  __/| |  | |\ V / (_| | ||  __/
    #  |_|   |_|  |_| \_/ \__,_|\__\___|

    def dynamics(self):
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

    def _load_racetrack(self, track: TrackData) -> dict:
        abscissa = track.track.abscissa

        # Create the splines
        spline = lambda name, values: ca.interpolant(
            name, "bspline", [abscissa], values
        )
        return {
            "s_values": abscissa,
            "x": spline("x", track.track.x_mid_line),
            "y": spline("y", track.track.y_mid_line),
            "z": spline("z", track.track.elevation),
            "theta": spline("theta", track.track.dir_mid_line),
            "banking": spline("banking", track.track.banking),
            "slope": spline("slope", track.track.slope),
            "rho": spline("rho", track.track.curvature),
            "n_l": spline("n_l", track.track.width_no_kerbs_L),
            "n_r": spline("n_r", -track.track.width_no_kerbs_R),
            "x_L": spline("x_L", track.track.x_margin_no_kerb_L),
            "y_L": spline("y_L", track.track.y_margin_no_kerb_L),
            "z_L": spline("z_L", track.track.z_margin_no_kerb_L),
            "x_R": spline("x_R", track.track.x_margin_no_kerb_R),
            "y_R": spline("y_R", track.track.y_margin_no_kerb_R),
            "z_R": spline("z_R", track.track.z_margin_no_kerb_R),
        }

    def _get_vehicle_data(self, data: dict) -> dict:
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
