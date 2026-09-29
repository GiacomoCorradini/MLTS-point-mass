# %% Setup
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from MLTS import GGVConstr

data_dir = os.path.join(os.path.dirname(__file__), "vehicle")

# %% Main
if __name__ == "__main__":

    # GGV from boundary samples (columns V, ax, ay)
    ggv_data = pd.read_csv(os.path.join(data_dir, "ggv_envelope.csv"))
    ggv = GGVConstr(ggv_data)
    ggv_scaled = GGVConstr(ggv_data, scales=[0.7])  # e.g. lower grip

    # GGV from FWBW tables (JSON)
    with open(os.path.join(data_dir, "ggv_envelope.json"), "r") as f:
        ggv_fwbw = GGVConstr(json.load(f))

    # 3D envelope
    ggv.plot()
    ggv_fwbw.plot()

    # G-G diagram at a few speeds
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))
    for ax, v in zip(axs, [10, 30, 50, 65]):
        ax.plot(*ggv.boundary(v), "r-", label="CSV")
        ax.plot(*ggv_scaled.boundary(v), "r--", label="CSV x 0.7")
        ax.plot(*ggv_fwbw.boundary(v), "b-", label="FWBW (JSON)")
        ax.set_title(f"V = {v} m/s")
        ax.set_xlabel(r"$a_y (m/s^2)$")
        ax.set_aspect("equal")
        ax.grid(True)
    axs[0].set_ylabel(r"$a_x (m/s^2)$")
    axs[0].legend()
    fig.tight_layout()

    plt.show()
