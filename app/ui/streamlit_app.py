"""Minimal Streamlit UI for Telecom BSS RAG Assistant."""

import requests
import streamlit as st

from app.logger import get_logger


logger = get_logger(__name__)


# ============================================================
# CONFIG
# ============================================================

API_URL = "http://127.0.0.1:8000/query"
HEALTH_URL = "http://127.0.0.1:8000/"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Telecom BSS Assistant (Built with Publicly Available Data)",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MINIMAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background-color: #0b0f17;
    }

    .main .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 6rem;
    }


    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] button {
        border-radius: 10px;
        min-height: 44px;
    }


    /* Chat spacing */
    [data-testid="stChatMessage"] {
        padding-top: 12px;
        padding-bottom: 12px;
    }

    [data-testid="stChatMessage"] .stMarkdown {
        font-size: 1rem;
        line-height: 1.65;
    }


    /* Larger avatars */
    [data-testid="stChatMessage"] [data-testid="chatAvatarIcon"] {
        width: 52px !important;
        height: 52px !important;
        min-width: 52px !important;
        min-height: 52px !important;
        border-radius: 50% !important;
    }


    /* More space between avatar and text */
    [data-testid="stChatMessage"] {
        column-gap: 16px !important;
    }


    /* Chat input */
    [data-testid="stChatInput"] {
        padding-bottom: 12px;
    }

    [data-testid="stChatInput"] > div {
        background-color: #171e2c;
        border: 1px solid #39445a;
        border-radius: 14px;
    }

    [data-testid="stChatInput"] textarea {
        color: #f8fafc !important;
        font-size: 1rem !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #7d899e !important;
    }


    /* Hide unnecessary Streamlit decoration */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BACKEND
# ============================================================

def check_backend() -> bool:
    """Check whether FastAPI backend is available."""

    try:
        response = requests.get(
            HEALTH_URL,
            timeout=3,
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


def ask_api(question: str) -> str:
    """Send question to FastAPI backend."""

    try:
        response = requests.post(
            API_URL,
            json={
                "question": question
            },
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        if data.get("status") == "error":
            raise RuntimeError(
                data.get(
                    "answer",
                    "Backend returned an error.",
                )
            )

        answer = data.get("answer")

        if not answer:
            raise RuntimeError(
                "Backend returned an empty answer."
            )

        return answer

    except Exception:
        logger.exception(
            "FastAPI request failed"
        )
        raise


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "quick_question" not in st.session_state:
    st.session_state.quick_question = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📚 BSS Knowledge")

    st.write("📡 Business Support Systems")
    st.write("💰 Rating & Charging")
    st.write("🧾 Billing & Invoicing")
    st.write("📦 Products & Subscriptions")
    st.write("💳 Payments & Revenue")

    st.divider()

    st.subheader("💡 Quick Questions")

    quick_questions = [
        "What is BSS?",
        "What is rating?",
        "What is charging?",
        "What is billing?",
        "What is a recurring charge?",
    ]

    for index, quick_question in enumerate(
        quick_questions
    ):

        if st.button(
            quick_question,
            key=f"quick_question_{index}",
            use_container_width=True,
        ):

            st.session_state.quick_question = (
                quick_question
            )

            st.rerun()

    st.divider()

    # --------------------------------------------------------
    # BACKEND STATUS
    # --------------------------------------------------------

    if check_backend():
        st.success("🟢 Backend online")
    else:
        st.error("🔴 Backend offline")

    st.divider()

    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear conversation",
        key="clear_chat",
        use_container_width=True,
    ):

        st.session_state.messages = []
        st.session_state.quick_question = None

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title("📡 Telecom BSS Assistant (Built with Publicly Available Data)")

st.caption(
    "RAG-powered knowledge assistant · Telecom BSS"
)

st.divider()


# ============================================================
# WELCOME
# ============================================================

if not st.session_state.messages:

    st.markdown(
        "### 👋 Welcome to the Telecom BSS Assistant"
    )

    st.write(
        "Ask questions about **BSS, rating, charging, "
        "billing, subscriptions, payments, and revenue "
        "management.**"
    )

    st.write("")


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        with st.chat_message(
            "user",
            avatar="👤",
        ):

            st.markdown(
                message["content"]
            )

    else:

        with st.chat_message(
            "assistant",
            avatar="🤖",
        ):

            st.markdown(
                message["content"]
            )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a Telecom BSS question..."
)


# ============================================================
# QUICK QUESTION
# ============================================================

if st.session_state.quick_question:

    question = st.session_state.quick_question

    st.session_state.quick_question = None


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    logger.info(
        "Processing question: %s",
        question,
    )

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message(
        "user",
        avatar="👤",
    ):

        st.markdown(question)


    # --------------------------------------------------------
    # CHECK BACKEND
    # --------------------------------------------------------

    if not check_backend():

        answer = (
            "The FastAPI backend is currently unavailable. "
            "Please make sure the FastAPI server is running "
            "on port 8000."
        )

        with st.chat_message(
            "assistant",
            avatar="🤖",
        ):

            st.error(answer)


    else:

        # ----------------------------------------------------
        # CALL FASTAPI
        # ----------------------------------------------------

        with st.chat_message(
            "assistant",
            avatar="🤖",
        ):

            with st.spinner(
                "🔎 Searching BSS knowledge..."
            ):

                try:

                    answer = ask_api(
                        question
                    )

                    st.markdown(answer)

                    logger.info(
                        "Assistant response received"
                    )

                except Exception:

                    logger.exception(
                        "Assistant request failed"
                    )

                    answer = (
                        "I couldn't connect to the "
                        "Telecom BSS backend. Please "
                        "check the FastAPI server."
                    )

                    st.error(answer)


    # --------------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
