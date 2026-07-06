"""
사용자 질문 → 벡터DB 검색 → Claude 호출 → 출처 포함 답변 반환
"""

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import config

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""당신은 대한민국 세법 전문가입니다.
반드시 아래 세법 조문만을 근거로 답변하세요.
조문에 없는 내용은 추측하지 말고 "해당 내용은 제공된 조문에서 확인되지 않습니다"라고 답하세요.
답변 마지막에 근거 조문을 명시하세요.

[관련 세법 조문]
{context}

[질문]
{question}

[답변]""",
)


def get_chain():
    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    db = Chroma(
        persist_directory=config.DB_DIR,
        embedding_function=embeddings,
    )
    llm = ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": config.TOP_K}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )
    return chain


def ask(question: str) -> dict:
    chain = get_chain()
    result = chain.invoke({"query": question})

    # 중복 출처 제거
    seen = set()
    sources = []
    for doc in result["source_documents"]:
        m = doc.metadata
        key = (m.get("law"), m.get("article_num"))
        if key not in seen:
            seen.add(key)
            sources.append({
                "law": m.get("law", ""),
                "article_num": m.get("article_num", ""),
                "article_title": m.get("article_title", ""),
                "effective_date": m.get("effective_date", ""),
            })

    return {
        "answer": result["result"],
        "sources": sources,
    }
