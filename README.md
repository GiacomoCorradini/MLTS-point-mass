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

`sol` (dict, arrays sampled on the mesh along `s`):

| Field | Description |
| --- | --- |
| `success`, `status`, `iterations`, `solve_time` | IPOPT outcome (last iterate returned if not converged) |
| `abscissa` | curvilinear abscissa `s` of the centreline [m] |
| `distance` | distance travelled along the trajectory [m] |
| `time` | lap time [s] |
| `n` | lateral offset from the centreline [m] |
| `Xi` | heading relative to the centreline [rad] |
| `kappa` | centreline curvature [1/m] |
| `kappa_trj` | trajectory curvature `ay / V²` [1/m] |
| `yaw_rate` | `ay / V` [rad/s] |
| `V`, `ax`, `ay` | speed [m/s], longitudinal and lateral acceleration [m/s²] |
| `x_trj`, `y_trj`, `z_trj` | optimal trajectory [m] |
| `x_mid_line`, `y_mid_line`, `z_mid_line` | centreline [m] |
| `x_margin_L/R`, `y_margin_L/R`, `z_margin_L/R` | left / right track borders [m] |

## Inputs

- **track**: path of the track file (tab separated, `#` comments) or `DataFrame` with columns `abscissa`, `curvature`, `dir_mid_line`, `x_mid_line`, `y_mid_line`, `width_no_kerbs_L`, `width_no_kerbs_R`; optional `elevation`, `slope` (positive uphill), `banking` (positive: right side up), default 0.
- **vehicle** (dict): `W_total` width [m], `v_max` [m/s]; optional `v_min` (5.0), `tau_ax`, `tau_ay` (0.03 s).
- **ggv**, either:
  - CSV / `DataFrame` with columns `V, ax, ay`: boundary samples at each speed (symmetric in `ay`);
  - FWBW dict: tables `ax_max(ay, V)`, `ax_min(ay, V)`, `ay_max(V)`, see [ggv_envelope.json](examples/vehicle/ggv_envelope.json).
- **ggv_scales**: grip scaling `[mu]`, `[mu_ax, mu_ay]` or `[mu_ax_max, mu_ax_min, mu_ay]`.

The GGV constraint is `(rho / r)^2 <= 1`, with `rho` the distance of `(ax, ay)` from the diagram centre and `r` that of the boundary in the same direction.
