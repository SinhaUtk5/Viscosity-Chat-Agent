"""Shared dead-oil viscosity prediction logic for MCP and Streamlit."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import xgboost as xgb


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("VISCOSITY_MODEL_PATH", APP_DIR / "Viscosity_XGB_L1out.json"))

_booster: xgb.Booster | None = None


def get_booster() -> xgb.Booster:
    """Load the model once per process."""
    global _booster
    if _booster is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Set VISCOSITY_MODEL_PATH or "
                "place Viscosity_XGB_L1out.json beside predictor.py."
            )
        _booster = xgb.Booster()
        _booster.load_model(MODEL_PATH)
    return _booster


def validate_inputs(temperature_c: float, molecular_weight: float, api_gravity: float) -> None:
    values = {
        "temperature_c": temperature_c,
        "molecular_weight": molecular_weight,
        "api_gravity": api_gravity,
    }
    for name, value in values.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")

    if temperature_c <= 0:
        raise ValueError("temperature_c must be greater than 0 degrees Celsius.")
    if molecular_weight <= 0:
        raise ValueError("molecular_weight must be greater than 0 g/mol.")
    if api_gravity <= 0:
        raise ValueError("api_gravity must be greater than 0.")
    if np.isclose(api_gravity, 1.0):
        raise ValueError("api_gravity cannot be 1 because log10(API) is used as a divisor.")


def build_feature_row(
    temperature_c: float,
    molecular_weight: float,
    api_gravity: float,
) -> np.ndarray:
    """Reproduce the feature engineering used by the original application."""
    validate_inputs(temperature_c, molecular_weight, api_gravity)

    sg = 141.5 / (131.5 + api_gravity)
    kw = 4.5579 * molecular_weight**0.15178 * sg**-0.84573
    log_api = np.log10(api_gravity)
    log_t = np.log10(temperature_c)
    mult = np.log10(molecular_weight) / log_api

    if not np.isfinite(mult):
        raise ValueError(
            "This API gravity produces an undefined engineered feature; "
            "please provide a different positive API gravity."
        )

    return np.asarray(
        [[temperature_c, kw, molecular_weight, sg, api_gravity, log_api, log_t, mult]],
        dtype=float,
    )


def predict_viscosity(
    temperature_c: float,
    molecular_weight: float,
    api_gravity: float,
) -> float:
    """Run the trained model and return dead-oil viscosity in cP."""
    features = build_feature_row(temperature_c, molecular_weight, api_gravity)
    raw_prediction = float(get_booster().predict(xgb.DMatrix(features))[0])
    viscosity_cp = float(10.0 ** (10.0**raw_prediction) - 1.0)
    if not np.isfinite(viscosity_cp):
        raise ValueError("The model returned a non-finite viscosity prediction.")
    return viscosity_cp
