# %% Setup
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pytelemsys.pytrack import TrackData
from MLTS import MLTS

data_dir = os.path.join(os.path.dirname(__file__), "vehicle_1")


def load_dict(path: str) -> dict:
    if path.endswith(".csv"):
        import pandas as pd

        return pd.read_csv(path)
    with open(path, "r") as f:
        if path.endswith((".yaml", ".yml")):
            import yaml

            return yaml.safe_load(f)
        return json.load(f)


# %% Main
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Minimum lap time of a point mass.")
    parser.add_argument("track", type=str, help="Path to circuit_data.txt.")
    parser.add_argument(
        "--ggv", type=str, default=os.path.join(data_dir, "ggv_envelope.json")
    )
    parser.add_argument(
        "--vehicle", type=str, default=os.path.join(data_dir, "vehicle.json")
    )
    parser.add_argument("--step", type=float, default=1.0, help="Mesh step [m].")
    args = parser.parse_args()

    track_data = TrackData(args.track)
    ggv_data = load_dict(args.ggv)
    vehicle_data = load_dict(args.vehicle)

    mlts = MLTS(track_data, vehicle_data, ggv_data)

    # Initial guess (n, Xi, V, ax, ay): constant state
    x0 = np.array([0.0, 0.0, 30.0, 0.0, 0.0])
    sol = mlts.solution(x0, step_size=args.step)

    fig = plt.figure()
    ax = fig.add_subplot(121, projection="3d")
    ax.plot(
        track_data.track.x_mid_line,
        track_data.track.y_mid_line,
        track_data.track.elevation,
        "k--",
        linewidth=1,
        label="Track centerline",
    )
    ax.plot(
        sol["x_trj"],
        sol["y_trj"],
        sol["z_trj"],
        "b-",
        linewidth=3,
        label="MLTS trajectory",
    )
    ax.plot(sol["x_L"], sol["y_L"], sol["z_L"], "k-", label="Track left kerb")
    ax.plot(sol["x_R"], sol["y_R"], sol["z_R"], "k-", label="Track right kerb")
    ax.set_aspect("equal", "datalim")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True)
    ax.legend()

    ax = fig.add_subplot(222)
    ax.plot(sol["s"], sol["n"], label="MLTS lateral deviation")
    ax.plot(
        mlts.track_data["s_values"],
        mlts.track_data["n_l"](mlts.track_data["s_values"]),
        "k-",
        linewidth=1,
        label="Left track boundary",
    )
    ax.plot(
        mlts.track_data["s_values"],
        mlts.track_data["n_r"](mlts.track_data["s_values"]),
        "k-",
        linewidth=1,
        label="Right track boundary",
    )
    ax.set_xlabel("s (m)")
    ax.set_ylabel("n (m)")
    ax.grid(True)
    ax.legend()

    ax = fig.add_subplot(224)
    ax.plot(sol["s"], sol["Xi"])
    ax.set_xlabel("s (m)")
    ax.set_ylabel(r"$\Xi$ (rad)")
    ax.grid(True)

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(sol["s"], sol["V"])
    ax.set_xlabel("s (m)")
    ax.set_ylabel("V (m/s)")
    ax.grid(True)

    ax = fig.add_subplot(122, projection="3d")
    v_range = np.linspace(mlts.ggv.v_min, mlts.ggv.v_max, 30)
    AY, AX = map(np.array, zip(*(mlts.ggv.boundary(v) for v in v_range)))
    V = np.repeat(v_range[:, None], AY.shape[1], axis=1)
    ax.plot_wireframe(
        AY,
        AX,
        V,
        rstride=2,
        cstride=10,
        color="red",
        linewidth=0.5,
        alpha=0.5,
        label="GGV constraint",
    )
    ax.plot(sol["ay"], sol["ax"], sol["V"], "b-", label="MLTS")
    ax.legend()
    ax.set_xlabel(r"$a_y (m/s^2)$")
    ax.set_ylabel(r"$a_x (m/s^2)$")
    ax.set_zlabel(r"V (m/s)")
    ax.set_title("G-G-V Diagram")
    ax.grid(True)

    plt.show()
