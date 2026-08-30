import sys
from pathlib import Path

# Make the project folder visible to Streamlit
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.generator import answer_question
from src.config import settings

st.set_page_config(page_title="RAG Assistant", page_icon="📚")

st.title("📚 Production-Style RAG Assistant")
st.caption(f"Model: {settings.groq_chat_model}")

question = st.text_input(
    "Ask a question about your documents:",
    placeholder="What is the remote work policy?",
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Searching documents and generating an answer..."):
        try:
            answer = answer_question(question)
            st.markdown("### Answer")
            st.write(answer)
        except Exception as error:
            st.error(str(error))