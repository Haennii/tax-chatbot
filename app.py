import streamlit as st
from pathlib import Path
import config

st.set_page_config(
    page_title="세법 챗봇",
    page_icon="⚖️",
    layout="centered",
)

st.title("⚖️ 세법 챗봇")
st.caption("법인세법 · 소득세법 · 부가가치세법 조문을 기반으로 답변합니다.")

# 벡터DB 존재 여부 확인
db_ready = Path(config.DB_DIR).exists() and any(Path(config.DB_DIR).iterdir())

if not db_ready:
    st.error(
        "세법 데이터가 준비되지 않았습니다.\n\n"
        "터미널에서 아래 순서로 실행하세요:\n\n"
        "```\n"
        "python -m src.fetcher    # 법제처에서 조문 다운로드\n"
        "python -m src.indexer   # 벡터DB 구축\n"
        "```"
    )
    st.stop()

# 챗봇 로직은 DB가 준비된 경우에만 import
from src.chatbot import ask

if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 히스토리 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📖 근거 조문"):
                for s in msg["sources"]:
                    st.write(f"- {s['law']} 제{s['article_num']}조 {s['article_title']} (시행 {s['effective_date']})")

# 입력창
if question := st.chat_input("세법 질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("관련 조문 검색 중..."):
            result = ask(question)

        st.write(result["answer"])

        if result["sources"]:
            with st.expander("📖 근거 조문"):
                for s in result["sources"]:
                    st.write(f"- {s['law']} 제{s['article_num']}조 {s['article_title']} (시행 {s['effective_date']})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
