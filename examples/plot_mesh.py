# %% Setup
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import MLTS.define_mesh as mesh
from MLTS import read_track

tracks_dir = os.path.join(os.path.dirname(__file__), "tracks")

# %% Main
if __name__ == "__main__":

    track_file = (
        sys.argv[1] if len(sys.argv) > 1 else os.path.join(tracks_dir, "Catalunya.txt")
    )
    track = read_track(track_file)
    s, kappa = track.abscissa.to_numpy(), track.curvature.to_numpy()

    # Uniform mesh
    uniform, _ = mesh.build_uniform_spatial_mesh(s, step_size=5.0)

    # Dense mesh close to start and end, coarse elsewhere
    dense, _ = mesh.build_dense_start_end_mesh(
        s, coarse_step=5.0, dense_step=1.0, dense_length=200.0
    )

    # Time-uniform mesh: cells of about dt seconds along a reference speed profile
    mesh_cfg = {
        "dt": 0.05,  # target cell duration [s]
        "hmax": 5.0,  # max cell length [m]
        "maxacc": 10.0,  # max acceleration [m/s^2]
        "minacc": -30.0,  # max deceleration [m/s^2]
        "vmin": 20.0,  # min reference speed [m/s] (optional)
        "vmax": 80.0,  # max reference speed [m/s] (optional)
        "curvature_velocity": [  # |kappa| [1/m] -> reference speed [m/s]
            {"k": 0.0, "v": 90.0},
            {"k": 0.005, "v": 60.0},
            {"k": 0.02, "v": 35.0},
            {"k": 0.05, "v": 20.0},
        ],
    }
    time_uniform, _ = mesh.build_time_uniform_mesh(s, kappa, mesh_cfg)

    # Mesh nodes on the track
    meshes = {
        "Uniform": uniform,
        "Dense start/end": dense,
        "Time-uniform": time_uniform,
    }
    fig, axs = plt.subplots(1, len(meshes), figsize=(18, 6))
    for ax, (name, nodes) in zip(axs, meshes.items()):
        ax.plot(track.x_mid_line, track.y_mid_line, "k-", lw=0.5)
        ax.plot(
            np.interp(nodes, s, track.x_mid_line),
            np.interp(nodes, s, track.y_mid_line),
            "r.",
            ms=2,
        )
        ax.set_title(f"{name} ({len(nodes) - 1} cells)")
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()

    plt.show()
