import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
with st.spinner("Loading models and database..."):
    from rag.chain import answer

st.set_page_config(page_title="PostgreSQL 16 Docs Assistant", page_icon="🐘")
st.title("PostgreSQL 16 Docs Assistant")
st.caption("Ask questions about PostgreSQL 16 — SELECT, indexes, transactions, and more.")

# ── Session state init ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []       # {"role": ..., "content": ...}
if "sources" not in st.session_state:
    st.session_state.sources = {}        # message_index → list of source dicts

# ── Render existing chat history ───────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i in st.session_state.sources:
            srcs = st.session_state.sources[i]
            if srcs:
                with st.expander("📚 Sources"):
                    for s in srcs:
                        st.markdown(f"- [{s['name']}]({s['url']})")

# ── Handle new user input ──────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a PostgreSQL question..."):
    # Show and store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call chain with full history (excluding the message we just appended)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving..."):
            result = answer(
                query=prompt,
                chat_history=st.session_state.messages[:-1],
            )
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("📚 Sources"):
                for s in result["sources"]:
                    st.markdown(f"- [{s['name']}]({s['url']})")

    # Store assistant message and its sources
    assistant_index = len(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
    st.session_state.sources[assistant_index] = result["sources"]