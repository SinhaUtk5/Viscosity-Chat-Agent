"""Fast batched predictor for the IPB&F dead-oil viscosity model."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import xgboost as xgb

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "Viscosity_XGB_L1out.json"
TEMPERATURE_OFFSETS = np.array(
    [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
    dtype=float,
)
MINIMUM_FIT_TEMPERATURE_C = 18.0


@lru_cache(maxsize=1)
def load_booster() -> xgb.Booster:
    """Load the trained model once per application process."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file was not found: {MODEL_PATH}")

    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    booster.set_param({"nthread": max(1, os.cpu_count() or 1)})
    return booster


def _finite_positive(values: np.ndarray, name: str) -> None:
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"{name} must contain positive finite values.")


def predict_viscosity(
    temperature_c: float | np.ndarray,
    molecular_weight: float | np.ndarray,
    api_gravity: float | np.ndarray,
) -> float | np.ndarray:
    """Predict smoothed dead-oil viscosity for one or many input rows.

    Inputs are broadcast to a common shape. For every row, the model predicts
    at T-5,...,T-1 and T+1,...,T+5. Temperature/prediction pairs below 18 °C
    are excluded. A linear fit is performed in the model's transformed target
    space and evaluated at the requested temperature.
    """
    temperature, mw, api = np.broadcast_arrays(
        np.asarray(temperature_c, dtype=float),
        np.asarray(molecular_weight, dtype=float),
        np.asarray(api_gravity, dtype=float),
    )
    output_shape = temperature.shape

    temperature = temperature.reshape(-1)
    mw = mw.reshape(-1)
    api = api.reshape(-1)

    if np.any(~np.isfinite(temperature)):
        raise ValueError("Temperature must contain finite values.")

    _finite_positive(mw, "Molecular weight")
    _finite_positive(api, "API gravity")

    temperature_grid = temperature[:, None] + TEMPERATURE_OFFSETS[None, :]
    valid = temperature_grid >= MINIMUM_FIT_TEMPERATURE_C
    valid_counts = valid.sum(axis=1)

    if np.any(valid_counts < 2):
        rows = (np.flatnonzero(valid_counts < 2) + 1).tolist()
        raise ValueError(
            "At least two neighboring temperatures must be at or above "
            f"{MINIMUM_FIT_TEMPERATURE_C:g} °C. Check input row(s): {rows}."
        )

    grid_shape = temperature_grid.shape
    mw_grid = np.broadcast_to(mw[:, None], grid_shape)
    api_grid = np.broadcast_to(api[:, None], grid_shape)

    model_temperature = temperature_grid[valid]
    model_mw = mw_grid[valid]
    model_api = api_grid[valid]

    model_sg = 141.5 / (131.5 + model_api)
    model_kw = 4.5579 * model_mw**0.15178 * model_sg**-0.84573

    features = np.column_stack(
        (
            model_temperature,
            model_kw,
            model_mw,
            model_sg,
            model_api,
            np.log10(model_api),
            np.log10(model_temperature),
            np.log10(model_mw) / np.log10(model_api),
        )
    )

    # All rows and valid neighboring temperatures are evaluated in one model
    # call. inplace_predict avoids DMatrix construction overhead.
    transformed_valid = np.asarray(
        load_booster().inplace_predict(features),
        dtype=float,
    )

    transformed_grid = np.zeros(grid_shape, dtype=float)
    transformed_grid[valid] = transformed_valid

    x_grid = np.broadcast_to(TEMPERATURE_OFFSETS[None, :], grid_shape)
    sum_x = np.where(valid, x_grid, 0.0).sum(axis=1)
    sum_y = np.where(valid, transformed_grid, 0.0).sum(axis=1)
    mean_x = sum_x / valid_counts
    mean_y = sum_y / valid_counts

    dx = np.where(valid, x_grid - mean_x[:, None], 0.0)
    dy = np.where(valid, transformed_grid - mean_y[:, None], 0.0)
    denominator = np.sum(dx * dx, axis=1)

    if np.any(denominator <= 0):
        raise ValueError("The valid temperatures do not support a linear fit.")

    slope = np.sum(dx * dy, axis=1) / denominator

    # x=0 is the requested temperature because x uses T-neighbor offsets.
    transformed_at_temperature = mean_y - slope * mean_x
    viscosity_cp = 10.0 ** (10.0**transformed_at_temperature) - 1.0
    viscosity_cp = viscosity_cp.reshape(output_shape)

    if output_shape == ():
        return float(viscosity_cp)

    return viscosity_cp

