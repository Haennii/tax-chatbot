"""
조문 데이터를 청크로 분할하고 ChromaDB 벡터DB에 저장합니다.
실행: python -m src.indexer
"""

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.loader import load_laws
import config


def build_index():
    """벡터DB 구축 (기존 DB가 있으면 덮어씀)"""
    docs = load_laws()
    if not docs:
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"청크 생성: {len(chunks)}개")

    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.DB_DIR,
    )
    print(f"벡터DB 구축 완료 → {config.DB_DIR}/")


if __name__ == "__main__":
    build_index()
