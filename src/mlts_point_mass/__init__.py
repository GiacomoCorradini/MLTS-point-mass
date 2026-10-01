from importlib.metadata import version, PackageNotFoundError

from mlts_point_mass.ocp import MLTS
from mlts_point_mass.ggv_constr import GGVConstr
from mlts_point_mass.track import read_track

try:
    __version__ = version("MLTS-point-mass")
except PackageNotFoundError:  # not installed, e.g. run from the sources
    __version__ = "unknown"

__all__ = ["MLTS", "GGVConstr", "read_track"]
