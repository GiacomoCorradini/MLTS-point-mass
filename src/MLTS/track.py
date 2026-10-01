import numpy as np
import pandas as pd


def read_track(track: str | pd.DataFrame) -> pd.DataFrame:
    """Read the track data, the 3D columns (elevation, slope, banking) default to 0.

    :param track: path of the track file (tab separated, "#" comments) or DataFrame
        with columns abscissa, curvature, dir_mid_line, x_mid_line, y_mid_line,
        width_no_kerbs_L, width_no_kerbs_R.
    :return: track data.
    """

    if isinstance(track, pd.DataFrame):
        data = track.copy()
    else:
        data = pd.read_csv(track, sep="\t", comment="#")

    for col in ("elevation", "slope", "banking"):
        if col not in data:
            data[col] = 0.0

    return data


def darboux_to_cartesian(x_ref, y_ref, z_ref, theta_ref, bank_ref, slope_ref, n):
    """Convert Darboux coordinates to Cartesian coordinates.

    x, y, z are ENU (z up); the road frame is Rz(theta) Ry(-slope) Rx(-bank).

    :param x_ref: x coordinate of the reference point.
    :param y_ref: y coordinate of the reference point.
    :param z_ref: z coordinate of the reference point.
    :param theta_ref: heading angle of the reference point.
    :param bank_ref: bank angle of the reference point (positive: right side up).
    :param slope_ref: slope angle of the reference point (positive uphill).
    :param n: lateral distance from the reference point (positive to the left).
    :return: x, y, z coordinates in Cartesian system.
    """

    s_bank, c_bank = np.sin(bank_ref), np.cos(bank_ref)
    s_slope, c_slope = np.sin(slope_ref), np.cos(slope_ref)
    s_theta, c_theta = np.sin(theta_ref), np.cos(theta_ref)

    x = x_ref + n * (c_theta * s_slope * s_bank - s_theta * c_bank)
    y = y_ref + n * (s_theta * s_slope * s_bank + c_theta * c_bank)
    z = z_ref - n * c_slope * s_bank

    return x, y, z
