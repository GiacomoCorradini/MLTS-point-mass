import numpy as np
import pandas as pd
import casadi as ca


class GGVConstr:
    """
    GGV constraint g(ax, ay, V) <= 1: the accelerations must stay inside the
    G-G diagram at speed V.

    g = (rho / r)^2, with rho the distance of (ax, ay) from the centre of the
    G-G diagram and r the distance of the boundary, in the same direction.
    """

    def __init__(
        self, ggv: dict | pd.DataFrame, scales: np.ndarray | list | None = None
    ) -> None:
        """Build the GGV envelope.

        :param ggv: samples (V, ax, ay) or FWBW tables.
        :param scales: grip scales [mu], [mu_ax, mu_ay] or
            [mu_ax_max, mu_ax_min, mu_ay], defaults to 1.
        """

        if "longitudinal_acceleration" in ggv and "lateral_acceleration" in ggv:
            v_grid, ax, ay = self._boundary_from_fwbw(ggv)
        elif "V" in ggv and "ax" in ggv and "ay" in ggv:
            v_grid, ax, ay = self._boundary_from_samples(ggv)
        else:
            raise ValueError("Invalid GGV data format")

        # Scales
        mu_ax_max, mu_ax_min, mu_ay = self._get_scales(scales)
        ax = [np.where(a >= 0, a * mu_ax_max, a * mu_ax_min) for a in ax]
        ay = [a * mu_ay for a in ay]

        self.v_grid = v_grid
        self.v_min, self.v_max = float(v_grid[0]), float(v_grid[-1])
        self._build_envelope(ax, ay)

    #   ___                   _
    #  |_ _|_ __  _ __  _   _| |_
    #   | || '_ \| '_ \| | | | __|
    #   | || | | | |_) | |_| | |_
    #  |___|_| |_| .__/ \__,_|\__|
    #            |_|

    def _boundary_from_samples(
        self, ggv: dict | pd.DataFrame, glitch_ratio: float = 0.7
    ):
        """Samples with the same V: the boundary at that speed, sorted by angle."""

        df = pd.DataFrame(ggv)[["V", "ax", "ay"]].astype(float)
        df["V"] = df["V"].round(6)  # same speed up to round-off
        df["ay"] = df["ay"].abs()  # symmetric envelope
        df = df.drop_duplicates()  # e.g. the ay < 0 half, mirrored onto ay >= 0
        df["angle"] = np.arctan2(df["ax"], df["ay"])

        v_grid, ax_rows, ay_rows = [], [], []
        for v, samples in df.sort_values("angle").groupby("V"):
            if len(samples) < 3:
                raise ValueError(
                    f"GGV samples: at least 3 samples per speed needed (V = {v})"
                )
            ax, ay = self._remove_glitches(
                samples["ax"].to_numpy(), samples["ay"].to_numpy(), glitch_ratio
            )
            v_grid.append(v)
            ax_rows.append(ax)
            ay_rows.append(ay)

        return np.array(v_grid), ax_rows, ay_rows

    @staticmethod
    def _remove_glitches(ax: np.ndarray, ay: np.ndarray, ratio: float):
        """Replace samples well inside both neighbours (solver glitches) with their mean."""

        ax, ay = ax.copy(), ay.copy()
        r = np.hypot(ax, ay)
        j = 1 + np.flatnonzero((r[1:-1] < ratio * r[:-2]) & (r[1:-1] < ratio * r[2:]))
        ax[j] = (ax[j - 1] + ax[j + 1]) / 2
        ay[j] = (ay[j - 1] + ay[j + 1]) / 2
        return ax, ay

    def _boundary_from_fwbw(self, ggv: dict, n_ay: int = 41):
        """Tables ax_max(ay, V), ax_min(ay, V), ay_max(V): ax_min -> tip (ay_max, 0) -> ax_max."""

        lon, lat = ggv["longitudinal_acceleration"], ggv["lateral_acceleration"]
        ay_tab = np.array(lon["lateral_acceleration"])
        v_tab = np.array(lon["longitudinal_velocity"])
        ax_max_tab = np.array(lon["positive_longitudinal_acceleration"]).reshape(
            len(v_tab), len(ay_tab)
        )
        ax_min_tab = np.array(lon["negative_longitudinal_acceleration"]).reshape(
            len(v_tab), len(ay_tab)
        )
        v_lat = np.array(lat["longitudinal_velocity"])
        ay_max_lat = np.array(lat["positive_lateral_acceleration"])

        # Each table row on s = ay / ay_max in [0, 1]: cells outside the
        # envelope (stored as 0) replaced by the tip ax = 0 at s = 1, so the
        # boundary scales with ay_max(V) instead of snapping to the ay grid
        s = np.linspace(0.0, 1.0, n_ay)
        ay_max_tab = np.interp(v_tab, v_lat, ay_max_lat)
        v_rows, lower_rows, upper_rows = [], [], []
        for v, a, lo, up in zip(v_tab, ay_max_tab, ax_min_tab, ax_max_tab):
            k = (up - lo > 1e-3) & (ay_tab >= 0.0) & (ay_tab < a)
            if not k.any():  # no envelope (e.g. beyond top speed)
                continue
            s_k = np.append(ay_tab[k], a) / a
            # |ax| limits non-increasing with ay (removes solver glitches)
            lo_k = np.maximum.accumulate(np.append(lo[k], 0.0))
            up_k = np.minimum.accumulate(np.append(up[k], 0.0))
            v_rows.append(v)
            lower_rows.append(np.interp(s, s_k, lo_k))
            upper_rows.append(np.interp(s, s_k, up_k))
        v_rows, lower_rows, upper_rows = map(np.array, (v_rows, lower_rows, upper_rows))

        ax_rows, ay_rows, v_grid = [], [], []
        for v, ay_max in zip(v_lat, ay_max_lat):
            # Rows interpolated (linearly) at this speed
            lower = [np.interp(v, v_rows, col) for col in lower_rows.T]
            upper = [np.interp(v, v_rows, col) for col in upper_rows.T]

            # Skip degenerate speeds (envelope collapsed to a point)
            if ay_max < 1e-3 or upper[0] - lower[0] < 1e-3:
                continue
            v_grid.append(v)
            ax_rows.append(np.concatenate((lower, upper[::-1])))
            ay_rows.append(np.concatenate((s, s[::-1])) * ay_max)

        return np.array(v_grid), ax_rows, ay_rows

    @staticmethod
    def _get_scales(scales) -> tuple[float, float, float]:
        """[mu], [mu_ax, mu_ay] or [mu_ax_max, mu_ax_min, mu_ay]."""

        if scales is None:
            return 1.0, 1.0, 1.0
        scales = np.atleast_1d(scales).astype(float)
        if len(scales) == 1:
            return scales[0], scales[0], scales[0]
        if len(scales) == 2:
            return scales[0], scales[0], scales[1]
        if len(scales) == 3:
            return scales[0], scales[1], scales[2]
        raise ValueError(
            "scales must be [mu], [mu_ax, mu_ay] or [mu_ax_max, mu_ax_min, mu_ay]"
        )

    #   _____                 _
    #  | ____|_ ____   _____| | ___  _ __   ___
    #  |  _| | '_ \ \ / / _ \ |/ _ \| '_ \ / _ \
    #  | |___| | | \ V /  __/ | (_) | |_) |  __/
    #  |_____|_| |_|\_/ \___|_|\___/| .__/ \___|
    #                               |_|

    def _build_envelope(
        self,
        ax: list[np.ndarray],
        ay: list[np.ndarray],
        n_psi: int = 73,
        n_mirror: int = 6,
        n_dense: int = 1000,
        eps: float = 1e-2,
    ):
        """ax, ay: boundary of each speed of v_grid, from pure braking to pure traction."""

        v_grid = self.v_grid
        ax_c = np.array([(a[0] + a[-1]) / 2 for a in ax])
        psi_grid = np.linspace(-np.pi / 2, np.pi / 2, n_psi)
        r_mat = np.zeros((n_psi, len(v_grid)))
        for i in range(len(v_grid)):
            # Densify the boundary along its polyline: interpolating r linearly
            # in psi between sparse samples bulges the envelope outwards (a
            # straight segment is r ~ 1/cos(psi - psi0), convex in psi)
            arc = np.concatenate(
                ([0.0], np.cumsum(np.hypot(np.diff(ax[i]), np.diff(ay[i]))))
            )
            new = np.concatenate(([True], np.diff(arc) > 0))
            arc_dense = np.linspace(0.0, arc[-1], n_dense)
            ax_i = np.interp(arc_dense, arc[new], ax[i][new])
            ay_i = np.interp(arc_dense, arc[new], ay[i][new])

            d = ax_i - ax_c[i]
            psi = np.arctan2(d, ay_i)
            # Star-shaped around the centre: keep the samples with increasing psi
            keep = np.concatenate(([True], psi[1:] > np.maximum.accumulate(psi)[:-1]))
            r_mat[:, i] = np.interp(psi_grid, psi[keep], np.hypot(d, ay_i)[keep])

        # Mirror across psi = +-pi/2 (i.e. across ay = 0): with |ay| in psi,
        # dr/dpsi = 0 there keeps g smooth in pure braking / traction
        dpsi = psi_grid[1] - psi_grid[0]
        k = np.arange(1, n_mirror + 1)
        psi_ext = np.concatenate(
            (psi_grid[0] - k[::-1] * dpsi, psi_grid, psi_grid[-1] + k * dpsi)
        )
        r_ext = np.concatenate((r_mat[k[::-1]], r_mat, r_mat[-1 - k]), axis=0)

        self.r_sp = ca.interpolant(
            "r", "bspline", [psi_ext, v_grid], r_ext.ravel(order="F")
        )
        self.ax_c_sp = ca.interpolant("ax_c", "bspline", [v_grid], ax_c)

        # g(ax, ay, V), held at the edge values outside the speed grid
        q = ca.MX.sym("q", 3)
        v_sat = ca.fmin(ca.fmax(q[2], v_grid[0]), v_grid[-1])
        d = q[0] - self.ax_c_sp(v_sat)
        psi = ca.atan2(d, ca.sqrt(q[1] ** 2 + eps**2))
        self.g = ca.Function(
            "g", [q], [(d**2 + q[1] ** 2) / self.r_sp(ca.vertcat(psi, v_sat)) ** 2]
        )

    def boundary(self, v: float, n: int = 181) -> tuple[np.ndarray, np.ndarray]:
        """Closed envelope boundary (ay, ax) at speed v."""

        v = float(np.clip(v, self.v_min, self.v_max))
        psi = np.linspace(-np.pi / 2, np.pi / 2, n)
        r = np.array(self.r_sp(np.vstack((psi, np.full(n, v))))).squeeze()
        ax = float(self.ax_c_sp(v)) + r * np.sin(psi)
        ay = r * np.cos(psi)
        return np.concatenate((ay, -ay[::-1])), np.concatenate((ax, ax[::-1]))

    #   ____  _       _
    #  |  _ \| | ___ | |_
    #  | |_) | |/ _ \| __|
    #  |  __/| | (_) | |_
    #  |_|   |_|\___/ \__|

    def plot(self, n_v: int = 40) -> None:
        """Plot the 3D GGV envelope, sampled at n_v speeds."""
        import matplotlib.pyplot as plt

        v_range = np.linspace(self.v_min, self.v_max, n_v)
        AY, AX = zip(*(self.boundary(v) for v in v_range))
        V = np.repeat(v_range[:, None], len(AY[0]), axis=1)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(np.array(AY), V, np.array(AX), cmap="coolwarm", alpha=0.8)
        ax.set_title("G-G-V Diagram")
        ax.set_xlabel(r"$a_y (m/s^2)$")
        ax.set_ylabel(r"V (m/s)")
        ax.set_zlabel(r"$a_x (m/s^2)$")
