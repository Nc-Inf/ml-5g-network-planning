import numpy as np

def log_distance_pl_db(distance_m: np.ndarray, pl0_db: float = 32.4, n: float = 2.2) -> np.ndarray:
    """
    Simple log-distance path loss proxy:
    PL(d) = PL(1m) + 10*n*log10(d)
    """
    d = np.maximum(distance_m, 1.0)
    return pl0_db + 10.0 * n * np.log10(d)

def received_power_dbm(tx_dbm: float, distance_m: np.ndarray, n: float = 2.2) -> np.ndarray:
    pl = log_distance_pl_db(distance_m, n=n)
    return tx_dbm - pl
