"""Streamlit interface for the Telecom BSS Knowledge Assistant."""

import streamlit as st
from app.retrieval.pipeline import ask, build_teleco_assistant
from app.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="Telecom BSS Assistant",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .welcome-box {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background: #f8fafc;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #6b7280;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 0.5rem;
    }
    .status-box {
        padding: 0.8rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown(
    '<div class="main-title">📡 Telecom BSS Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered knowledge assistant for Telecom Business Support Systems</div>',
    unsafe_allow_html=True
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("📚 BSS Knowledge")
    st.markdown(
        """
        This assistant can help with concepts such as:

        - Business Support Systems (BSS)
        - Charging & Rating
        - Billing & Invoicing
        - Payments & Balances
        - Recurring & Usage Charges
        - Discounts & Taxes
        - Collections
        - Revenue Assurance
        - Reconciliation
        """
    )

    st.divider()
    st.subheader("💡 Try asking")

    suggestions = [
        "What is billing?",
        "What is a recurring charge?",
        "What is charging?",
        "What is the difference between billing and payment?",
        "What is revenue assurance?"
    ]

    for suggestion in suggestions:
        if st.button(suggestion, use_container_width=True):
            logger.info("Sidebar suggestion selected: %s", suggestion)
            st.session_state.pending_question = suggestion

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        logger.info("Conversation history cleared")
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Telecom BSS Knowledge Assistant")
    st.caption("Grounded responses • No unsupported answers")

# ---------- Agent ----------
@st.cache_resource(show_spinner="Setting up Telecom BSS Assistant...")
def get_agent():
    """Initialize and cache the Telecom BSS assistant."""
    try:
        logger.info("Initializing Telecom BSS assistant for Streamlit")
        agent = build_teleco_assistant()
        logger.info("Telecom BSS assistant initialized successfully")
        return agent
    except Exception:
        logger.exception("Failed to initialize Telecom BSS assistant")
        raise

agent = get_agent()

# ---------- Session State ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ---------- Welcome ----------
if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-box">
            <h4>👋 Welcome!</h4>
            <p>
                Ask a question about Telecom BSS concepts and I'll answer
                using the available Telecom knowledge base.
            </p>
            <p>
                <b>Example:</b> What is billing?
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------- Chat History ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------- Question ----------
question = st.chat_input("Ask a question about Telecom BSS...")

# Handle sidebar suggestion
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

# ---------- Process Question ----------
if question:
    logger.info("Processing user question: %s", question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the Telecom BSS knowledge base..."):
            try:
                answer = ask(agent, question)
                logger.info("Assistant response generated successfully")
            except Exception:
                logger.exception("Failed to generate response for user question")
                raise

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

