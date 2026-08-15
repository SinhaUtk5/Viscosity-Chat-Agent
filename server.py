"""MCP server for the IPB&F dead-oil viscosity XGBoost model.

The chat host supplies the conversational agent. Because temperature, molecular
weight, and API gravity are required tool inputs, the agent asks for any values
the user did not provide before calling the predictor.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from predictor import predict_viscosity
mcp = FastMCP(
    "IPBF Dead Oil Viscosity",
    instructions=(
        "Use predict_dead_oil_viscosity when the user wants dead-oil viscosity. "
        "Temperature in degrees Celsius, stock-tank-oil molecular weight in "
        "g/mol, and API gravity are all required. Never guess a missing value: "
        "ask a short follow-up question for only the missing input or inputs, "
        "then call the tool once all three are available."
    ),
)


@mcp.tool(
    title="Predict dead-oil viscosity",
    description=(
        "Predict dead-oil viscosity from exactly three required values: "
        "temperature in degrees Celsius, stock-tank-oil molecular weight in "
        "g/mol, and API gravity. If the user has not supplied all three, do not "
        "call this tool and do not infer defaults; ask for only the missing values."
    ),
)
def predict_dead_oil_viscosity(
    temperature_c: float,
    molecular_weight: float,
    api_gravity: float,
) -> dict[str, float | str]:
    """Run the trained XGBoost model and return viscosity in cP."""
    viscosity_cp = predict_viscosity(temperature_c, molecular_weight, api_gravity)

    return {
        "temperature_c": float(temperature_c),
        "molecular_weight_g_mol": float(molecular_weight),
        "api_gravity": float(api_gravity),
        "viscosity_cp": viscosity_cp,
        "display": f"Predicted dead-oil viscosity: {viscosity_cp:.4g} cP",
        "method": "IPB&F XGBoost dead-oil viscosity model",
    }


if __name__ == "__main__":
    # ChatGPT/Codex connects to the streamable HTTP MCP endpoint exposed here.
    mcp.run(transport="streamable-http")
