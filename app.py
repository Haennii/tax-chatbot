import streamlit as st
from pathlib import Path
import config

st.set_page_config(
    page_title="세법 챗봇",
    page_icon="⚖️",
    layout="centered",
)

st.title("⚖️ 세법 챗봇")
st.caption("법인세법 · 소득세법 · 부가가치세법 · 조세특례제한법 조문을 기반으로 답변합니다.")

db_ready = Path(config.DB_DIR).exists() and any(Path(config.DB_DIR).iterdir())

if not db_ready:
    st.error(
        "세법 데이터가 준비되지 않았습니다.\n\n"
        "터미널에서 아래 순서로 실행하세요:\n\n"
        "```\n"
        "python -m src.fetcher\n"
        "python -m src.indexer\n"
        "```"
    )
    st.stop()

from src.chatbot import ask_stream

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📖 근거 조문"):
                for s in msg["sources"]:
                    st.write(f"- {s['law']} 제{s['article_num']}조 {s['article_title']} (시행 {s['effective_date']})")

if question := st.chat_input("세법 질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        full_text = ""
        sources = []
        placeholder = st.empty()

        for chunk, src in ask_stream(question):
            if src is not None:
                sources = src
            full_text += chunk
            placeholder.markdown(full_text + "▌")

        placeholder.markdown(full_text)

        if sources:
            with st.expander("📖 근거 조문"):
                for s in sources:
                    st.write(f"- {s['law']} 제{s['article_num']}조 {s['article_title']} (시행 {s['effective_date']})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_text,
        "sources": sources,
    })
