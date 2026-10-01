# %% Setup
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from mlts_point_mass import GGVConstr

data_dir = os.path.join(os.path.dirname(__file__))

# %% Main
if __name__ == "__main__":

    # GGV from CSV
    ggv_data = pd.read_csv(os.path.join(data_dir, "vehicle_3", "ggv_envelope.csv"))
    ggv_csv = GGVConstr(ggv_data)

    # GGV from JSON
    with open(os.path.join(data_dir, "vehicle_2", "ggv_envelope.json"), "r") as f:
        ggv_data = json.load(f)

    ggv_json = GGVConstr(ggv_data)
    ggv_json_scaled = GGVConstr(ggv_data, scales=[0.7])  # e.g. lower grip

    # 3D envelope
    ggv_json.plot()
    ggv_json_scaled.plot()

    # G-G diagram at a few speeds
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))
    for ax, v in zip(axs, [10, 30, 50, 65]):
        ax.plot(*ggv_csv.boundary(v), "b-", label="Original (CSV)")
        ax.plot(*ggv_json.boundary(v), "g-", label="Original (JSON)")
        ax.plot(*ggv_json_scaled.boundary(v), "r--", label="Scaled (JSON 70%)")
        ax.set_title(f"V = {v} m/s")
        ax.set_xlabel(r"$a_y (m/s^2)$")
        ax.set_aspect("equal")
        ax.grid(True)
    axs[0].set_ylabel(r"$a_x (m/s^2)$")
    axs[0].legend()
    fig.tight_layout()

    plt.show()
