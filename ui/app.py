"""
Streamlit UI for Finance Invoice RAG Bot

Features:
- Upload invoices for indexing
- Ask questions about your invoices
- See grounding scores and source documents
- View retrieval quality metrics
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.domain_chunker import MultiDocChunker
from core.hybrid_search import HybridSearchEngine
from core.llm_pipeline import LLMPipeline


st.set_page_config(
    page_title="Finance Invoice Q&A Bot",
    page_icon="invoice",
    layout="wide",
)

st.title("Finance Invoice Q&A Bot")
st.caption("RAG-powered question answering over invoice documents")

# Initialize session state
if "search_engine" not in st.session_state:
    st.session_state.search_engine = HybridSearchEngine()
if "pipeline" not in st.session_state:
    st.session_state.pipeline = LLMPipeline(st.session_state.search_engine)
if "chunker" not in st.session_state:
    st.session_state.chunker = MultiDocChunker()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: Document Upload
with st.sidebar:
    st.header("Documents")
    st.info(f"Indexed chunks: {len(st.session_state.search_engine.chunks)}")

    uploaded_files = st.file_uploader(
        "Upload invoice text files",
        type=["txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        all_chunks = []
        for file in uploaded_files:
            text = file.read().decode("utf-8")
            chunks = st.session_state.chunker.chunk_document(text, "invoice", file.name)
            all_chunks.extend(chunks)

        if st.button("Index Documents"):
            with st.spinner("Building search index..."):
                st.session_state.search_engine.build_index(all_chunks)
            st.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} files")

    # Sample data
    st.divider()
    st.subheader("Sample Invoices")
    if st.button("Load Sample Data"):
        from scripts.generate_dataset import generate_sample_invoices
        sample_invoices = generate_sample_invoices(5)
        all_chunks = []
        for inv in sample_invoices:
            chunks = st.session_state.chunker.chunk_document(inv, "invoice", "sample")
            all_chunks.extend(chunks)
        st.session_state.search_engine.build_index(all_chunks)
        st.success(f"Loaded {len(all_chunks)} chunks from 5 sample invoices")

# Main area: Chat
st.header("Ask a Question")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.write(f"**{src.get('type', 'unknown')}** (score: {src.get('score', 0):.4f})")
                    st.write(src.get("content", "")[:200])
                    if src.get("metadata"):
                        st.json(src["metadata"])

# Input
question = st.chat_input("Ask about your invoices...")

if question:
    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Get response
    with st.chat_message("assistant"):
        if not st.session_state.search_engine.is_built:
            st.warning("No documents indexed yet. Upload invoices in the sidebar first.")
        else:
            with st.spinner("Searching and generating answer..."):
                response = st.session_state.pipeline.query(question)

            st.write(response.answer)

            # Show grounding score
            if response.retrieval_used:
                score = response.groundedness_score
                color = "green" if score >= 0.7 else "orange" if score >= 0.5 else "red"
                st.caption(f"Groundedness: :{color}[{score:.0%}] | Sources: {len(response.sources)}")

                if response.unsupported_claims:
                    st.warning(f"Could not verify: {', '.join(response.unsupported_claims[:3])}")

            # Save to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources,
            })

# Stats panel
with st.expander("Pipeline Statistics"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Chunks Indexed", len(st.session_state.search_engine.chunks))
    with col2:
        st.metric("Index Built", "Yes" if st.session_state.search_engine.is_built else "No")
    with col3:
        st.metric("Embedding Model", st.session_state.search_engine.embedding_model_name)
