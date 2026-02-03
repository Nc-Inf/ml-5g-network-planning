import numpy as np

def make_grid(area_m: float, grid_step_m: float) -> np.ndarray:
    """Return grid points (N,2) for a square area centered at (0,0)."""
    half = area_m / 2.0
    xs = np.arange(-half, half + grid_step_m, grid_step_m)
    ys = np.arange(-half, half + grid_step_m, grid_step_m)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])
