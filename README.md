# MLTS-point-mass

Minimum lap time of a point mass with a G-G-V constraint, solved as an OCP with [CasADi](https://web.casadi.org/) and IPOPT.

## Installation

```bash
pip install -e .
```

## Usage

```bash
python examples/run_mlts.py examples/tracks/Catalunya.txt   # --ggv, --vehicle, --step, --save sol.csv
python examples/plot_ggv.py
```

```python
from MLTS import MLTS

mlts = MLTS(track, vehicle, ggv, ggv_scales=None)
sol = mlts.solution(x0=[0, 0, 30, 0, 0], step_size=1.0)  # initial guess [n, Xi, V, ax, ay]
```

`sol`: solver outcome (`success`, `status`, `iterations`, `solve_time`), states and `time` along `s`, trajectory (`x_trj`, `y_trj`, `z_trj`).

## Inputs

- **track**: `pytelemsys.pytrack.TrackData`.
- **vehicle** (dict): `W_total` width [m], `v_max` [m/s]; optional `v_min` (5.0), `tau_ax`, `tau_ay` (0.03 s).
- **ggv**, either:
  - CSV / `DataFrame` with columns `V, ax, ay`: boundary samples at each speed (symmetric in `ay`);
  - FWBW dict: tables `ax_max(ay, V)`, `ax_min(ay, V)`, `ay_max(V)`, see [ggv_envelope.json](examples/vehicle/ggv_envelope.json).
- **ggv_scales**: grip scaling `[mu]`, `[mu_ax, mu_ay]` or `[mu_ax_max, mu_ax_min, mu_ay]`.

The GGV constraint is `(rho / r)^2 <= 1`, with `rho` the distance of `(ax, ay)` from the diagram centre and `r` that of the boundary in the same direction.
