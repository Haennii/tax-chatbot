"""
사용자 질문 → BM25 키워드 검색 → Ollama 호출 → 출처 포함 답변 반환
"""

import json
from pathlib import Path
from rank_bm25 import BM25Okapi
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
import config

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a Korean tax law expert. You must always respond in Korean (한국어).

아래 세법 조문을 근거로 질문에 답하세요.
반드시 한국어로만 답변하세요.
조문에 없는 내용은 "해당 내용은 제공된 조문에서 확인되지 않습니다"라고 답하세요.
답변 마지막에 근거 조문을 명시하세요.

[관련 세법 조문]
{context}

[질문]
{question}

[한국어 답변]""",
)


def load_all_articles():
    articles = []
    for path in Path(config.LAWS_DIR).glob("*.json"):
        with open(path, encoding="utf-8") as f:
            articles.extend(json.load(f))
    return articles


def bm25_search(query: str, articles: list, k: int = 5) -> list:
    corpus = [a["text"] for a in articles]
    tokenized = [list(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(list(query))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [articles[i] for i in top_indices]


def ask(question: str) -> dict:
    articles = load_all_articles()
    top_articles = bm25_search(question, articles, k=config.TOP_K)

    context = "\n\n".join(
        f"[{a['law']} 제{a['article_num']}조 {a['article_title']}]\n{a['text']}"
        for a in top_articles
    )

    llm = ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL)
    prompt_text = PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt_text)

    sources = [
        {
            "law": a["law"],
            "article_num": a["article_num"],
            "article_title": a["article_title"],
            "effective_date": a["effective_date"],
        }
        for a in top_articles
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }
