# IPB&F Dead-Oil Viscosity MCP Server

This package provides two interfaces around `Viscosity_XGB_L1out.json`:

1. A GPT-powered Streamlit agent that understands natural conversation, asks
   for missing inputs, and calls the local XGBoost model through function calling.
2. An MCP Streamable HTTP server for ChatGPT/Codex tool calls.

Example conversation:

> **User:** Predict viscosity at 60 °C and API 32.  
> **Agent:** What is the stock-tank-oil molecular weight in g/mol?  
> **User:** 225  
> **Agent:** Predicted dead-oil viscosity: ... cP

## Files

Place the trained model beside `server.py`:

```text
viscosity_mcp/
├── predictor.py
├── gpt_agent.py
├── server.py
├── streamlit_app.py
├── requirements.txt
└── Viscosity_XGB_L1out.json
```

Alternatively, set `VISCOSITY_MODEL_PATH` to its full path.

## Configure the OpenAI API key

Create an API key in the OpenAI Platform. Do not paste it into Python code or
commit it to GitHub.

For the current PowerShell window:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
$env:OPENAI_MODEL="gpt-5.6-luna"  # optional
```

For Streamlit Community Cloud, open **App settings → Secrets** and add:

```toml
OPENAI_API_KEY = "your_api_key_here"
OPENAI_MODEL = "gpt-5.6-luna"
```

## Run the GPT-powered Streamlit chat UI

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

GPT interprets natural language and maintains the conversation through the
Responses API. The `predict_dead_oil_viscosity` function schema requires all
three inputs, so GPT asks only for missing values and calls the trained local
model after the inputs are complete.

Optional logos are loaded automatically when `TAMU.png` and `IPBF.png` are placed
beside `streamlit_app.py`.

## Host on Streamlit Community Cloud

1. Add these files and `Viscosity_XGB_L1out.json` to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set the entrypoint to `streamlit_app.py`.
4. Add `OPENAI_API_KEY` under **App settings → Secrets**.
5. Deploy.

Keep the model repository private if its trained weights should not be public.

## Run the MCP server separately

```bash
python server.py
```

The default MCP endpoint is normally `http://localhost:8000/mcp`. Test it with:

```bash
npx @modelcontextprotocol/inspector@latest
```

Choose **Streamable HTTP** and connect to the endpoint printed by the server.

## Connect to ChatGPT

ChatGPT must reach an HTTPS URL. During development, expose the local port with
a secure tunnel, then add `https://YOUR_HOST/mcp` as a custom MCP server in
developer mode. For production, deploy the same process behind a stable HTTPS
endpoint.

The Streamlit chat requires an OpenAI API key because GPT provides the agent
behavior. The trained viscosity calculation remains local to your application;
only the conversation and structured function-call inputs are sent through the
OpenAI API.

Streamlit Community Cloud hosts the Streamlit web application, not a general
MCP Streamable HTTP endpoint. Deploy `server.py` on a service that supports a
long-running HTTP process if you want ChatGPT/Codex to connect to `/mcp`.

## Why missing inputs trigger a follow-up

All three parameters in `predict_dead_oil_viscosity` are required by the MCP
input schema. Its tool description and server instructions also tell the agent
not to guess. Therefore a request such as “T = 50 °C, API = 35” remains in the
chat until the agent obtains molecular weight, after which it invokes the tool.

## Important model-range note

The server enforces only the mathematical conditions required by the current
feature equations. Add minimum and maximum limits from the model's actual
training data before public deployment so the tool can reject extrapolation.
