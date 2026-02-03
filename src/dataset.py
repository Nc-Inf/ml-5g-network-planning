import numpy as np
from .users import users_hotspots
from .baselines import kmeans_bs
from .geometry import make_grid
from .coverage import coverage_score

def scenario_features(users_xy: np.ndarray, n_bs: int) -> np.ndarray:
    """
    Scenario-level features independent of BS placement (prevents leakage).
    """
    center = users_xy.mean(axis=0)
    user_dist = np.linalg.norm(users_xy - center, axis=1)

    features = [
        float(n_bs),                      # number of base stations
        float(len(users_xy)),             # number of users
        float(user_dist.mean()),          # avg dispersion
        float(user_dist.std()),           # spread
        float(np.ptp(users_xy[:, 0]) + np.ptp(users_xy[:, 1]))  # span proxy
    ]
    return np.array(features, dtype=float)

def generate_dataset(
    n_samples: int,
    area_m: float,
    grid_step: float,
    tx_dbm: float,
    thr_dbm: float,
    seed: int = 0
):
    rng = np.random.default_rng(seed)
    grid = make_grid(area_m, grid_step)

    X, y = [], []

    for i in range(n_samples):
        n_users = int(rng.integers(200, 600))
        n_bs = int(rng.integers(3, 10))

        users = users_hotspots(n_users, area_m, n_hotspots=3, seed=1000 + i)

        # placement strategy used to generate target (KMeans baseline)
        bs = kmeans_bs(users, n_bs, seed=2000 + i)

        score = coverage_score(grid, bs, tx_dbm=tx_dbm, thr_dbm=thr_dbm)
        feats = scenario_features(users, n_bs)

        X.append(feats)
        y.append(score)

    return np.array(X), np.array(y)
