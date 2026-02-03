import numpy as np

def log_distance_pl_db(distance_m: np.ndarray, f_hz: float, pl0_db: float = 32.4, n: float = 2.0) -> np.ndarray:
    """
    Simple log-distance path loss:
    PL(d) = PL(d0=1m) + 10*n*log10(d)
    pl0_db default ~32.4 dB at 1 GHz & 1 m (rough). Keep as proxy.
    """
    d = np.maximum(distance_m, 1.0)
    return pl0_db + 10.0 * n * np.log10(d)

def received_power_dbm(tx_dbm: float, distance_m: np.ndarray, f_hz: float, n: float = 2.0) -> np.ndarray:
    pl = log_distance_pl_db(distance_m, f_hz=f_hz, n=n)
    return tx_dbm - pl
