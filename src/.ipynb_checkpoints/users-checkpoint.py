import numpy as np

def users_uniform(n_users: int, area_m: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = area_m / 2.0
    return rng.uniform(-half, half, size=(n_users, 2))

def users_hotspots(
    n_users: int,
    area_m: float,
    n_hotspots: int = 3,
    seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = area_m / 2.0
    centers = rng.uniform(-half, half, size=(n_hotspots, 2))
    idx = rng.integers(0, n_hotspots, size=n_users)
    users = centers[idx] + rng.normal(0, area_m * 0.05, size=(n_users, 2))
    return np.clip(users, -half, half)
