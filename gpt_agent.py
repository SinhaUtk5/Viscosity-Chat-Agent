"""GPT conversation and function-calling loop for the viscosity calculator."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from predictor import predict_viscosity

SYSTEM_INSTRUCTIONS = """
You are the IPB&F Dead-Oil Viscosity Assistant.

Your only calculation tool predicts dead-oil viscosity. Every calculation row
requires:
1. temperature in degrees Celsius,
2. stock-tank-oil molecular weight in g/mol, and
3. API gravity.

Accept either one input set or several input sets. Users may provide multiple
sets as natural-language lists, arrays, matrices, or pasted rows. For an
unlabelled numeric matrix, interpret each row in this fixed order only:
[temperature_c, molecular_weight, api_gravity]. If the order is ambiguous, ask
the user to confirm it instead of guessing.

Retain values supplied earlier in the conversation. Never invent a missing
value. When a value clearly applies to every row—for example, one molecular
weight and API with several temperatures—expand it across those rows. If any
row remains incomplete, ask one short question identifying the missing field
and row. Do not calculate only the complete subset unless the user explicitly
asks you to.

As soon as all rows are complete, call predict_dead_oil_viscosity exactly once
with all rows in samples. The local tool performs the trained XGBoost
calculation and T-neighborhood smoothing. Do not perform or approximate the
calculation yourself.

After the tool returns:
- For one row, report the viscosity in cP and repeat the three inputs used.
- For multiple rows, return a concise Markdown table with row number,
  temperature, molecular weight, API gravity, and viscosity in cP.
- Preserve the input row order.
- If the tool returns an error, explain it briefly and request corrected input.
""".strip()


VISCOSITY_TOOL = {
    "type": "function",
    "name": "predict_dead_oil_viscosity",
    "description": (
        "Run the trained IPB&F XGBoost model with T-neighborhood smoothing "
        "for one or more complete input rows. Submit every row in one call."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "samples": {
                "type": "array",
                "description": (
                    "One or more viscosity input rows, retained in the same "
                    "order supplied by the user."
                ),
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "temperature_c": {
                            "type": "number",
                            "description": "Temperature in degrees Celsius.",
                        },
                        "molecular_weight": {
                            "type": "number",
                            "description": (
                                "Stock-tank-oil molecular weight in g/mol."
                            ),
                        },
                        "api_gravity": {
                            "type": "number",
                            "description": "Stock-tank-oil API gravity.",
                        },
                    },
                    "required": [
                        "temperature_c",
                        "molecular_weight",
                        "api_gravity",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["samples"],
        "additionalProperties": False,
    },
}


def _as_float_vector(values: list[Any], name: str) -> np.ndarray:
    """Convert tool inputs into a finite one-dimensional float array."""
    vector = np.asarray(values, dtype=float)

    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")

    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains a missing or non-finite value.")

    return vector


def execute_viscosity_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate all rows and execute one vectorized predictor call."""
    samples = arguments.get("samples")

    if not isinstance(samples, list) or not samples:
        raise ValueError("At least one complete input row is required.")

    temperatures = _as_float_vector(
        [sample["temperature_c"] for sample in samples],
        "Temperature",
    )
    molecular_weights = _as_float_vector(
        [sample["molecular_weight"] for sample in samples],
        "Molecular weight",
    )
    api_gravities = _as_float_vector(
        [sample["api_gravity"] for sample in samples],
        "API gravity",
    )

    if np.any(molecular_weights <= 0):
        raise ValueError("Molecular weight must be greater than zero.")

    if np.any(api_gravities <= 0):
        raise ValueError("API gravity must be greater than zero.")

    # The updated predictor accepts arrays and performs one batched XGBoost
    # prediction for every row and its valid T±1,...,T±5 neighborhood.
    predicted = predict_viscosity(
        temperatures,
        molecular_weights,
        api_gravities,
    )
    viscosities = np.asarray(predicted, dtype=float).reshape(-1)

    if viscosities.size != temperatures.size:
        raise ValueError(
            "The predictor returned a different number of results than inputs."
        )

    if not np.all(np.isfinite(viscosities)):
        raise ValueError("The predictor returned a non-finite viscosity.")

    results = [
        {
            "row": index + 1,
            "temperature_c": float(temperature),
            "molecular_weight_g_mol": float(molecular_weight),
            "api_gravity": float(api_gravity),
            "viscosity_cp": float(viscosity),
        }
        for index, (temperature, molecular_weight, api_gravity, viscosity) in enumerate(
            zip(
                temperatures,
                molecular_weights,
                api_gravities,
                viscosities,
                strict=True,
            )
        )
    ]

    return {
        "status": "success",
        "number_of_results": len(results),
        "results": results,
    }


def run_agent_turn(
    client: Any,
    user_text: str,
    model: str,
    previous_response_id: str | None = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Run one conversational turn, including requested local tool calls."""
    request: dict[str, Any] = {
        "model": model,
        "instructions": SYSTEM_INSTRUCTIONS,
        "input": user_text,
        "tools": [VISCOSITY_TOOL],
        "tool_choice": "auto",
        # One batched tool call is faster and keeps result ordering deterministic.
        "parallel_tool_calls": False,
        "reasoning": {"effort": "low"},
    }

    if previous_response_id:
        request["previous_response_id"] = previous_response_id

    response = client.responses.create(**request)
    tool_results: list[dict[str, Any]] = []

    for _ in range(3):
        tool_calls = [item for item in response.output if item.type == "function_call"]

        if not tool_calls:
            text = response.output_text.strip()
            if not text:
                text = "I could not produce a response. Please try that message again."
            return text, response.id, tool_results

        tool_outputs: list[dict[str, str]] = []

        for call in tool_calls:
            try:
                if call.name != "predict_dead_oil_viscosity":
                    raise ValueError(f"Unsupported tool: {call.name}")

                result = execute_viscosity_tool(json.loads(call.arguments))
                tool_results.append(result)

            except Exception as exc:
                result = {
                    "status": "error",
                    "message": str(exc),
                }

            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result),
                }
            )

        response = client.responses.create(
            model=model,
            instructions=SYSTEM_INSTRUCTIONS,
            previous_response_id=response.id,
            input=tool_outputs,
            tools=[VISCOSITY_TOOL],
            tool_choice="auto",
            parallel_tool_calls=False,
            reasoning={"effort": "low"},
        )

    raise RuntimeError("The agent exceeded the allowed number of tool-call rounds.")
