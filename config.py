from dotenv import load_dotenv
import os

load_dotenv()

# API 키
MOLEG_API_KEY = os.getenv("MOLEG_API_KEY")   # 법제처 Open API 아이디

# 경로
DATA_DIR = "data"
LAWS_DIR = "data/laws"
CASES_DIR = "data/cases"
DB_DIR = "db"

# 모델 설정 (Ollama 로컬 모델)
LLM_MODEL = "llama3"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

# RAG 설정
CHUNK_SIZE = 500        # 문서를 자르는 단위 (글자 수)
CHUNK_OVERLAP = 50      # 청크 간 겹치는 부분
TOP_K = 5               # 검색 시 가져올 문서 수
