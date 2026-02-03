"""
Baseline regression model for 5G coverage prediction.

This module provides a simple but interpretable baseline
to study the relationship between base-station placement
and coverage quality.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


def generate_synthetic_data(
    n_samples: int = 1000,
    random_state: int = 42
):
    """
    Generate a synthetic dataset representing simplified
    5G coverage behavior.

    Features:
    - distance_to_bs (m)
    - bs_density (sites / km^2)
    - transmit_power (dBm)

    Target:
    - coverage_quality (arbitrary units)
    """
    rng = np.random.default_rng(random_state)

    distance = rng.uniform(50, 2000, n_samples)
    density = rng.uniform(1, 10, n_samples)
    tx_power = rng.uniform(30, 46, n_samples)

    # Simplified coverage proxy
    coverage = (
        tx_power
        - 20 * np.log10(distance)
        + 5 * np.log10(density)
        + rng.normal(0, 2, n_samples)
    )

    X = np.column_stack([distance, density, tx_power])
    y = coverage

    return X, y


def train_baseline_model():
    X, y = generate_synthetic_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return model, mse


if __name__ == "__main__":
    model, mse = train_baseline_model()
    print(f"Baseline MSE: {mse:.2f}")
