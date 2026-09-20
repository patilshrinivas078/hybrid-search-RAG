"""
Streamlit UI for the RAG pipeline.

Two pages, selected from a sidebar (not st.tabs - see note below):
  - Ask Questions   : chat interface, wraps the existing handle_query()
  - Upload Documents: adds a new PDF/TXT/DOCX permanently to the knowledge
                       base via HybridRAGSearch.add_document()

Model loading (embedder, reranker, vectorstore) happens once via
st.cache_resource, not on every rerun. Streamlit reruns this whole script
on every interaction, so without caching you'd reload the reranker on every
click.

Usage:
    streamlit run streamlit_app.py
"""

import logging
import sys
import tempfile
from pathlib import Path

import streamlit as st
from langfuse import get_client

from src.hybrid_search import HybridRAGSearch
from src.reranker import CrossEncoderReranker
from src.generator import generate
from app import handle_query  # reuse the traced retrieve+generate wrapper as-is

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", stream=sys.stdout, force=True)


@st.cache_resource(show_spinner="Loading models and index (first run only)...")
def get_retriever() -> HybridRAGSearch:
    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3", device="cuda")
    return HybridRAGSearch(reranker=reranker)


st.set_page_config(page_title="Policy RAG", layout="wide")

st.sidebar.title("Policy RAG")
page = st.sidebar.radio("Go to", ["Ask Questions", "Upload Documents"])

retriever = get_retriever()

EXAMPLE_QUESTIONS = [
    "How much will I get if my two-wheeler bike's tyre is damaged, and does it matter whether I have the Tyre and Rim add-on?",
    "If a policyholder dies by suicide during an overseas trip, will their family receive any payout?",
    "What discount does the Anti-Theft device endorsement provide?",
]

st.title("Policy Document Assistant")

if page == "Ask Questions":
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.caption("Try asking:")
        cols = st.columns(len(EXAMPLE_QUESTIONS))
        for col, q in zip(cols, EXAMPLE_QUESTIONS):
            with col:
                if st.button(q, key=f"example_{q}", use_container_width=True):
                    st.session_state.queued_query = q

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Top-level call, not nested in a container - keeps it pinned to the
    # bottom of the viewport (see module docstring).
    typed_query = st.chat_input("Ask a question about your policy documents...")
    query = typed_query or st.session_state.pop("queued_query", None)
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = handle_query(retriever, query, top_k=4)
                except Exception as e:
                    answer = f"Something went wrong answering that: {e}"
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        # Streamlit's process stays alive between reruns (unlike the CLI),
        # but flushing after each answer keeps trace latency low and avoids
        # ever losing a trace if the app is stopped/redeployed mid-session.
        get_client().flush()

elif page == "Upload Documents":
    st.write("Upload a policy document to add it to the knowledge base permanently.")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx"])

    if uploaded_file is not None:
        if st.button("Add to knowledge base"):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Streamlit gives an in-memory buffer; add_document() needs a
                # real file path (PyPDFLoader/TextLoader/Docx2txtLoader all
                # read from disk), so write it to a temp file first.
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    result = retriever.add_document(tmp_path)
                except Exception as e:
                    st.error(f"Failed to add {uploaded_file.name}: {e}")
                else:
                    st.success(
                        f"Added **{result['filename']}** "
                        f"({result['chunks_added']} chunks, "
                        f"classified as `{result['policy_type']}`). "
                        f"Index now has {result['total_chunks_in_index']} chunks total."
                    )
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
                    get_client().flush()