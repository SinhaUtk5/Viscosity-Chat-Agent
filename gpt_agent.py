"""GPT conversation and function-calling loop for the viscosity calculator."""

from __future__ import annotations

import json
from typing import Any

from predictor import predict_viscosity


SYSTEM_INSTRUCTIONS = """
You are the IPB&F Dead-Oil Viscosity Assistant.

Your only calculation tool predicts dead-oil viscosity. To use it you must have:
1. temperature in degrees Celsius,
2. stock-tank-oil molecular weight in g/mol, and
3. API gravity.

Understand values from natural conversation and retain values supplied earlier
in the conversation. Never guess or invent a missing value. Ask a short,
specific follow-up question for only the missing value or values. As soon as all
three values are available, call predict_dead_oil_viscosity. After the tool
returns, report the viscosity in cP and clearly repeat the three inputs used.
If the tool returns an error, explain it briefly and ask for the corrected input.
Do not perform the viscosity calculation yourself.
""".strip()


VISCOSITY_TOOL = {
    "type": "function",
    "name": "predict_dead_oil_viscosity",
    "description": (
        "Run the trained IPB&F XGBoost model to predict dead-oil viscosity. "
        "Call only after the user has supplied all three required inputs."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "temperature_c": {
                "type": "number",
                "description": "Temperature of interest in degrees Celsius.",
            },
            "molecular_weight": {
                "type": "number",
                "description": "Stock-tank-oil molecular weight in g/mol.",
            },
            "api_gravity": {
                "type": "number",
                "description": "Stock-tank-oil API gravity.",
            },
        },
        "required": ["temperature_c", "molecular_weight", "api_gravity"],
        "additionalProperties": False,
    },
}


def execute_viscosity_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate tool arguments, execute the local model, and format its result."""
    temperature_c = float(arguments["temperature_c"])
    molecular_weight = float(arguments["molecular_weight"])
    api_gravity = float(arguments["api_gravity"])
    viscosity_cp = predict_viscosity(temperature_c, molecular_weight, api_gravity)

    return {
        "status": "success",
        "temperature_c": temperature_c,
        "molecular_weight_g_mol": molecular_weight,
        "api_gravity": api_gravity,
        "viscosity_cp": viscosity_cp,
    }


def run_agent_turn(
    client: Any,
    user_text: str,
    model: str,
    previous_response_id: str | None = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Run one conversational turn, including any requested local tool call."""
    request: dict[str, Any] = {
        "model": model,
        "instructions": SYSTEM_INSTRUCTIONS,
        "input": user_text,
        "tools": [VISCOSITY_TOOL],
        "tool_choice": "auto",
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

        tool_outputs = []
        for call in tool_calls:
            try:
                if call.name != "predict_dead_oil_viscosity":
                    raise ValueError(f"Unsupported tool: {call.name}")
                result = execute_viscosity_tool(json.loads(call.arguments))
                tool_results.append(result)
            except Exception as exc:
                result = {"status": "error", "message": str(exc)}

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
