import numpy as np
from sklearn.cluster import KMeans

def random_bs(n_bs: int, area_m: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = area_m / 2.0
    return rng.uniform(-half, half, size=(n_bs, 2))

def grid_bs(n_bs: int, area_m: float) -> np.ndarray:
    side = int(np.ceil(np.sqrt(n_bs)))
    half = area_m / 2.0
    xs = np.linspace(-half, half, side)
    ys = np.linspace(-half, half, side)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    return pts[:n_bs]

def kmeans_bs(users_xy: np.ndarray, n_bs: int, seed: int = 0) -> np.ndarray:
    km = KMeans(n_clusters=n_bs, n_init=10, random_state=seed)
    km.fit(users_xy)
    return km.cluster_centers_
