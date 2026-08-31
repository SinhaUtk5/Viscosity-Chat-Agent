"""GPT-powered Streamlit chat UI for the IPB&F dead-oil viscosity model."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from openai import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
from PIL import Image

from gpt_agent import run_agent_turn

# =============================================================================
# Application configuration
# =============================================================================

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "gpt-5.6-luna"

st.set_page_config(
    page_title="ViscoSense AI—the IPB&F Dead-Oil Viscosity Agent.",
    page_icon="🛢️",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --maroon: #500000;
        --maroon-soft: rgba(80, 0, 0, 0.08);
        --gold: #D6A84B;
        --ink: #202124;
        --line: rgba(80, 0, 0, 0.16);
    }

    html, body, [class*="css"] {
        font-size: 16px;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    [data-testid="stChatMessage"] {
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 0.7rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 7px 22px rgba(30, 20, 20, 0.04);
    }

    .hero-card {
        background:
            radial-gradient(circle at 92% 12%, rgba(214, 168, 75, 0.22), transparent 30%),
            linear-gradient(135deg, rgba(80, 0, 0, 0.12), rgba(80, 0, 0, 0.025));
        border: 1px solid var(--line);
        border-top: 4px solid var(--maroon);
        border-radius: 20px;
        padding: 1.45rem 1.6rem;
        margin: 1rem 0 0.75rem;
        box-shadow: 0 12px 35px rgba(80, 0, 0, 0.07);
    }

    .hero-kicker {
        color: var(--maroon);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .hero-title {
        color: var(--ink);
        font-size: 1.35rem;
        font-weight: 750;
        margin-bottom: 0.45rem;
    }

    .hero-copy {
        color: #3f4145;
        line-height: 1.65;
        margin: 0;
        max-width: 900px;
    }

    .api-box {
        max-width: 720px;
        margin: 1rem auto;
        padding: 1.35rem 1.5rem;
        background: linear-gradient(145deg, #ffffff, rgba(80, 0, 0, 0.04));
        border: 1px solid var(--line);
        border-radius: 18px;
        box-shadow: 0 12px 35px rgba(80, 0, 0, 0.06);
    }

    .publication-box {
        background: rgba(128, 128, 128, 0.055);
        border-left: 4px solid var(--gold);
        border-radius: 12px;
        padding: 0.95rem 1.15rem;
        margin: 0.75rem 0 1rem;
        color: #45474b;
        line-height: 1.55;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(80, 0, 0, 0.10);
    }

    .stButton > button, .stLinkButton > a {
        border-radius: 12px;
        min-height: 2.75rem;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Helper functions
# =============================================================================


def get_setting(name: str, default: str | None = None) -> str | None:
    """Read an environment variable or Streamlit Cloud secret."""
    value = os.getenv(name)

    if value:
        return value

    try:
        return st.secrets.get(name, default)
    except (FileNotFoundError, KeyError):
        return default


def show_logo(filename: str, width: int = 200) -> None:
    """Display a logo if its file is available."""
    image_path = APP_DIR / filename

    if image_path.is_file():
        st.image(Image.open(image_path), width=width)


def show_resized_image(image_name: str, target_height: int = 200) -> None:
    """Display an image at a fixed height while preserving aspect ratio."""
    image_path = APP_DIR / image_name

    if not image_path.is_file():
        return

    image = Image.open(image_path)
    width, height = image.size

    if height == 0:
        return

    resized_width = int(width * target_height / height)
    resized_image = image.resize((resized_width, target_height))

    st.image(resized_image)


def reset_chat() -> None:
    """Reset the conversation while retaining the API key."""
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Welcome! I am the ViscoSense AI—the IPB&F Dead-Oil Viscosity Agent. "
                "I can predict dead-oil viscosity using the IPB&F's "
                "Dead Oil Viscosity model. Please provide the temperature, stock-tank-oil "
                "molecular weight, and API gravity. You may provide the values "
                "together or one at a time."
            ),
        }
    ]

    st.session_state.previous_response_id = None


def clear_api_key() -> None:
    """Remove the entered API key and return to the access screen."""
    st.session_state.pop("user_api_key", None)
    st.session_state.pop("messages", None)
    st.session_state.pop("previous_response_id", None)


def format_api_error(exc: Exception) -> str:
    """Convert API exceptions into clear user-facing messages."""
    if isinstance(exc, AuthenticationError):
        return (
            "The OpenAI API key was rejected. Click **Change API key**, "
            "verify the key, and try again."
        )

    if isinstance(exc, RateLimitError):
        error_text = str(exc).lower()

        quota_messages = (
            "insufficient_quota",
            "credit_balance_exhausted",
            "no credits remaining",
            "exceeded your current quota",
        )

        if any(message in error_text for message in quota_messages):
            return (
                "The OpenAI API account has no remaining credits. Add credits "
                "in the OpenAI Platform billing settings and try again. "
                "A ChatGPT subscription does not include API credits."
            )

        return (
            "The OpenAI API rate limit was reached. Please wait briefly and "
            "submit the request again."
        )

    if isinstance(exc, APIConnectionError):
        return (
            "The application could not connect to the OpenAI API. Check the "
            "internet connection and try again."
        )

    if isinstance(exc, APIError):
        return (
            "The OpenAI API could not complete the request. Please try again. "
            f"Technical detail: `{exc}`"
        )

    return (
        "The viscosity calculation could not be completed. Check the model "
        f"files and application configuration. Technical detail: `{exc}`"
    )


# =============================================================================
# Logos
# =============================================================================

left_space, logo_left, logo_right, right_space = st.columns([1, 1, 1, 1])

with logo_left:
    show_logo("TAMU.png", width=200)

with logo_right:
    show_logo("IPBF.png", width=200)


# =============================================================================
# API-key access screen
# =============================================================================

configured_api_key = get_setting("OPENAI_API_KEY")
model = get_setting("OPENAI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL

st.title("IPB&F Dead-Oil Viscosity AI Agent")
st.caption(
    " A product of the Interaction of Phase-Behavior and Flow (IPB&F) Consortium"
)

# Automatically use a configured environment/Streamlit secret.
if configured_api_key and "user_api_key" not in st.session_state:
    st.session_state.user_api_key = configured_api_key


if "user_api_key" not in st.session_state:
    st.markdown(
        """
        <div class="api-box">
        <h3>Connect to the Viscosity Agent</h3>
        Enter your OpenAI Platform API key to begin. The key is password-masked
        and retained only for the current Streamlit session.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("api_key_form"):
        entered_api_key = st.text_input(
            "OpenAI API key",
            type="password",
            placeholder="sk-...",
            help="Enter an OpenAI Platform API key with available API credits.",
        )

        open_agent = st.form_submit_button(
            "Open Viscosity Agent",
            type="primary",
            use_container_width=True,
        )

    st.caption(
        "Important: ChatGPT Plus, Pro, or Team subscriptions do not "
        "automatically include OpenAI API credits."
    )

    if open_agent:
        entered_api_key = entered_api_key.strip()

        if not entered_api_key:
            st.error("Please enter an OpenAI API key.")
        elif not entered_api_key.startswith("sk-"):
            st.error(
                "This does not appear to be a valid OpenAI API key. "
                "The key should normally begin with `sk-`."
            )
        else:
            st.session_state.user_api_key = entered_api_key
            reset_chat()
            st.rerun()

    # Nothing below this point is shown until the key is entered.
    st.stop()


api_key = st.session_state.user_api_key
client = OpenAI(api_key=api_key)


# =============================================================================
# Sidebar controls
# =============================================================================

with st.sidebar:
    st.header("Agent Settings")

    st.success("API key entered")
    st.caption(f"GPT model: `{model}`")

    if st.button("Change API key", use_container_width=True):
        clear_api_key()
        st.rerun()

    if st.button("Start new conversation", use_container_width=True):
        reset_chat()
        st.rerun()


# =============================================================================
# Agent description
# =============================================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-kicker">Research-powered calculation</div>
        <div class="hero-title">Dead-oil viscosity, through a simple conversation</div>
        <p class="hero-copy">
            Provide temperature, stock-tank-oil molecular weight, and API gravity.
            The agent identifies the values, asks for anything missing, runs the
            calculation, and reports viscosity in centipoise.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.link_button(
    "↗ Open the IPB&F Dead-Oil Viscosity Calculator",
    "https://biroldindorukeorpvtlab-viscosity-dead-oil.streamlit.app/",
    type="primary",
    use_container_width=True,
)
st.caption(
    "This agent runs the calculation available at the IPB&F Dead-Oil "
    "Viscosity Calculator linked above."
)

st.markdown("#### Research foundation")

st.markdown(
    """
    <div class="publication-box">
    Inspired by <strong>SPE-209610-PA</strong>: Sinha, U., Dindoruk, B., and
    Soliman, M. Y. (2022), <i>Physics-Augmented Correlations and Machine-Learning
    Methods to Accurately Calculate Dead-Oil Viscosity Based on the Available
    Inputs</i>, SPE Journal, 27(5), 3240–3253.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# =============================================================================
# Chat interface
# =============================================================================

chat_heading, reset_column = st.columns([5, 1])

with chat_heading:
    st.subheader("Ask the Viscosity Agent")
    st.caption(
        "Enter the properties naturally. The agent will ask for any "
        "required information that is missing."
    )

with reset_column:
    if st.button("↻ Start over", use_container_width=True):
        reset_chat()
        st.rerun()


if "messages" not in st.session_state:
    reset_chat()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input(
    "Example: temperature is 60 °C, molecular weight is 220, and API is 32"
)

if prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Interpreting the inputs and running the IPB&F model..."):
            try:
                response_text, response_id, _ = run_agent_turn(
                    client=client,
                    user_text=prompt,
                    model=model,
                    previous_response_id=(st.session_state.previous_response_id),
                )

                st.session_state.previous_response_id = response_id

            except (
                AuthenticationError,
                RateLimitError,
                APIConnectionError,
                APIError,
            ) as exc:
                response_text = format_api_error(exc)

            except Exception as exc:
                response_text = format_api_error(exc)

        st.markdown(response_text)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
        }
    )


# =============================================================================
# Faculty and contributor section
# =============================================================================

st.divider()
st.subheader("Research Team")

researcher_col, faculty_col = st.columns(2)

with researcher_col:
    show_resized_image("utkarsh.jpg", target_height=200)
    st.markdown("**Utkarsh Sinha**")
    st.caption("Volunteer Research Associate")

with faculty_col:
    show_resized_image("birol.jpg", target_height=200)
    st.markdown("**Dr. Birol Dindoruk**")
    st.caption("Faculty Advisor · IPB&F Consortium")
