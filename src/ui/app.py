"""
Enhanced Streamlit dashboard for Vault Copilot.
All data access goes through the API — no direct database imports.
"""
import os
import streamlit as st
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Vault Copilot", layout="wide", page_icon="🔒")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .metric-card {
        background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #58a6ff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        margin-top: 0.3rem;
    }
    .trace-step {
        background: #161b22;
        border-left: 3px solid #58a6ff;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
    }
    .trace-tool {
        color: #f0883e;
        font-weight: 600;
    }
    .success-badge {
        background: #238636;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
    }
    .warning-badge {
        background: #d29922;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🔒 Vault Copilot")
st.sidebar.markdown(
    "**v2.0 — Production Architecture**\n"
    "- Multi-step ReAct Agent\n"
    "- Hybrid RAG + Cross-Encoder\n"
    "- AST SQL Guardrails\n"
    "- Receipt Deduplication\n"
    "- Confidence Scoring\n"
    "- Zero PII Egress"
)

# API configuration
API_URL = os.getenv("VAULT_API_URL", "http://localhost:8000/api")
api_key = st.sidebar.text_input("API Key", type="password", value="", help="Leave empty if auth is disabled")

page = st.sidebar.radio(
    "Navigation",
    ["💬 Agentic Chat", "📊 Financial Intel", "🧾 Upload"],
)


def get_headers():
    """Build request headers with optional API key."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def api_request(method: str, endpoint: str, **kwargs):
    """Make an API request with error handling."""
    headers = get_headers()
    if "headers" in kwargs:
        headers.update(kwargs.pop("headers"))
    try:
        resp = requests.request(
            method,
            f"{API_URL}/{endpoint}",
            headers=headers,
            timeout=300,
            **kwargs,
        )
        if resp.status_code == 401:
            st.error("🔑 Authentication failed. Please check your API key.")
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("🔌 Backend offline. Start the API server with `uvicorn src.api.main:app`")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The AI models are running on CPU and may need more time.")
        return None
    except Exception as e:
        st.error(f"❌ Request failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Page: Agentic Chat
# ---------------------------------------------------------------------------
if page == "💬 Agentic Chat":
    st.title("🤖 Financial Copilot")
    st.caption("Multi-step reasoning across SQL, RAG, and analytics tools")

    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "traces" not in st.session_state:
        st.session_state.traces = []

    # Display chat history
    for i, msg in enumerate(st.session_state.chat):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            # Show trace for assistant messages
            if msg["role"] == "assistant" and i < len(st.session_state.traces):
                trace = st.session_state.traces[i]
                if trace:
                    with st.expander("🔍 Agent Execution Trace", expanded=False):
                        for step in trace:
                            st.markdown(
                                f'<div class="trace-step">'
                                f'<span class="trace-tool">Step {step.get("step_number", "?")}: '
                                f'{step.get("tool_selected", "?")}</span><br>'
                                f'<em>{step.get("reasoning", "")[:200]}</em><br>'
                                f'<small>⏱️ {step.get("latency_ms", 0):.0f}ms</small>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

    # Chat input
    if prompt := st.chat_input("E.g., 'What did I spend the most on, and are there anomalies?'"):
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agent is reasoning and executing tools..."):
                result = api_request("POST", "chat", json={"query": prompt})
                if result:
                    answer = result.get("response", "Error processing request.")
                    trace = result.get("execution_trace", [])
                    steps = result.get("steps_taken", 0)
                    latency = result.get("total_latency_ms", 0)

                    st.write(answer)

                    # Show execution metadata
                    col1, col2 = st.columns(2)
                    col1.caption(f"🔧 {steps} tool(s) used")
                    col2.caption(f"⏱️ {latency:.0f}ms total")

                    # Show trace
                    if trace:
                        with st.expander("🔍 Agent Execution Trace", expanded=True):
                            for step in trace:
                                st.markdown(
                                    f'<div class="trace-step">'
                                    f'<span class="trace-tool">Step {step.get("step_number", "?")}: '
                                    f'{step.get("tool_selected", "?")}</span><br>'
                                    f'<em>{step.get("reasoning", "")[:200]}</em><br>'
                                    f'<small>⏱️ {step.get("latency_ms", 0):.0f}ms</small>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                    st.session_state.chat.append({"role": "assistant", "content": answer})
                    st.session_state.traces.append(trace)
                else:
                    st.session_state.traces.append([])


# ---------------------------------------------------------------------------
# Page: Financial Intel
# ---------------------------------------------------------------------------
elif page == "📊 Financial Intel":
    st.title("📊 Deterministic Analytics")
    st.caption("Real-time spending analysis — no LLM involved")

    result = api_request("GET", "analytics")
    if result:
        # --- Summary Metrics Row ---
        summary = result.get("summary", "")
        anomalies_text = result.get("anomalies", "")
        subs_text = result.get("subscriptions", "")
        trends_text = result.get("trends", "")
        categories_data = result.get("categories", "")
        report = result.get("report", "")

        # Try to extract numbers from the summary for metrics
        if summary:
            st.subheader("📋 Summary")
            st.info(summary)

        col1, col2 = st.columns(2)

        with col1:
            if anomalies_text and anomalies_text != "No anomalies detected.":
                st.subheader("🚨 Anomalies")
                st.warning(anomalies_text)
            else:
                st.subheader("🚨 Anomalies")
                st.success("No anomalies detected.")

        with col2:
            if subs_text and "No" not in subs_text[:10]:
                st.subheader("🔄 Subscriptions")
                st.info(subs_text)
            else:
                st.subheader("🔄 Subscriptions")
                st.info("No recurring subscriptions detected yet.")

        if trends_text:
            st.subheader("📈 Spending Trends")
            st.write(trends_text)

        if categories_data:
            st.subheader("📦 Category Breakdown")
            st.write(categories_data)

        # Fallback: show raw report
        if not any([summary, anomalies_text, subs_text, trends_text]) and report:
            st.subheader("📋 Full Report")
            st.write(report)

    else:
        st.info("No data available. Upload some receipts to get started!")


# ---------------------------------------------------------------------------
# Page: Upload
# ---------------------------------------------------------------------------
elif page == "🧾 Upload":
    st.title("🧾 Secure Receipt Ingestion")
    st.caption("Upload receipt images for OCR extraction and dual-memory storage")

    file = st.file_uploader(
        "Upload Receipt Image",
        type=["png", "jpg", "jpeg"],
        help="Max 10MB. Supported formats: JPEG, PNG.",
    )

    if file and st.button("🔒 Process & Vault", type="primary"):
        with st.spinner("🧠 Processing receipt — OCR + AI extraction on CPU may take 60-90 seconds..."):
            # Build multipart form data
            files = {"file": (file.name, file.getvalue(), "image/jpeg")}
            headers = {}
            if api_key:
                headers["X-API-Key"] = api_key

            try:
                resp = requests.post(
                    f"{API_URL}/upload",
                    files=files,
                    headers=headers,
                    timeout=300,
                )

                if resp.status_code == 200:
                    data = resp.json()

                    if data.get("duplicate"):
                        st.warning("⚠️ " + data.get("message", "Duplicate detected"))
                    elif data.get("extracted_data", {}).get("extraction_failed"):
                        st.error("❌ " + data.get("message", "Extraction failed"))
                        with st.expander("Debug Info"):
                            st.json(data.get("extracted_data", {}))
                    else:
                        st.success("✅ " + data.get("message", "Receipt processed"))

                        # Show extracted data
                        extracted = data.get("extracted_data", {})
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Store", extracted.get("store", "Unknown"))
                        col2.metric("Total", f"${extracted.get('total', 0):.2f}")
                        col3.metric("Category", extracted.get("category", "Unknown"))

                        confidence = extracted.get("ocr_confidence", 0)
                        if extracted.get("low_confidence"):
                            st.warning(f"⚠️ Low OCR confidence: {confidence:.1%}")
                        else:
                            st.caption(f"OCR confidence: {confidence:.1%}")

                        with st.expander("Full Extraction Details"):
                            st.json(extracted)

                elif resp.status_code == 401:
                    st.error("🔑 Authentication failed. Check your API key.")
                elif resp.status_code == 413:
                    st.error("📦 File too large. Maximum size is 10MB.")
                else:
                    st.error(f"❌ Upload failed (HTTP {resp.status_code})")

            except requests.exceptions.ConnectionError:
                st.error("🔌 Backend offline. Start the API server first.")
            except Exception as e:
                st.error(f"❌ Error: {e}")