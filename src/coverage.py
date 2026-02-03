import numpy as np
from .propagation import received_power_dbm

def coverage_score(grid_xy: np.ndarray, bs_xy: np.ndarray, tx_dbm: float, f_hz: float, thr_dbm: float = -90.0) -> float:
    """
    Coverage score = fraction of grid points where best BS received power > threshold.
    """
    # distances: (Ngrid, Nbs)
    d = np.linalg.norm(grid_xy[:, None, :] - bs_xy[None, :, :], axis=2)
    pr = received_power_dbm(tx_dbm, d, f_hz=f_hz, n=2.2)
    best = pr.max(axis=1)
    return float((best > thr_dbm).mean())

def best_power_map(grid_xy: np.ndarray, bs_xy: np.ndarray, tx_dbm: float, f_hz: float) -> np.ndarray:
    d = np.linalg.norm(grid_xy[:, None, :] - bs_xy[None, :, :], axis=2)
    pr = received_power_dbm(tx_dbm, d, f_hz=f_hz, n=2.2)
    return pr.max(axis=1)
