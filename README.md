# 세법 챗봇 (Tax Law Chatbot)

세법 조문과 판례를 기반으로 답변하는 RAG(검색 증강 생성) 챗봇입니다.  
일반 LLM의 할루시네이션을 극복하기 위해, 실제 세법 데이터를 검색한 뒤 그 근거를 바탕으로만 답변합니다.

---

## 작동 원리

```
사용자 질문
    ↓
벡터DB에서 관련 세법 조문 / 판례 검색
    ↓
LLM에게 "이 자료를 근거로만 답변해" 지시
    ↓
출처가 명시된 신뢰도 높은 답변
```

---

## 기술 스택

| 역할 | 기술 |
|---|---|
| UI | Streamlit |
| LLM | Claude API (Anthropic) |
| 벡터DB | ChromaDB |
| RAG 프레임워크 | LangChain |
| 임베딩 | OpenAI text-embedding-3-small |

---

## 프로젝트 구조

```
tax-chatbot/
│
├── app.py              # Streamlit 앱 실행 진입점
├── config.py           # 모델명, 파라미터 등 설정값
├── .env                # API 키 (git 제외 - .gitignore 처리됨)
├── .env.example        # API 키 템플릿 (git 포함)
├── requirements.txt    # 패키지 목록
│
├── data/               # 세법 원본 파일 보관
│   ├── laws/           # 세법 조문 (PDF, TXT)
│   └── cases/          # 판례 (PDF, TXT)
│
├── db/                 # 벡터DB 저장소 (자동 생성, git 제외)
│
└── src/                # 내부 로직
    ├── loader.py       # data/ 파일 읽기 및 청크 분할
    ├── indexer.py      # 벡터DB 구축 및 업데이트
    └── chatbot.py      # 검색 + LLM 호출 + 답변 생성
```

---

## 설치 및 실행

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. API 키 설정

`.env.example`을 복사해 `.env` 파일을 만들고 키를 입력합니다.

```bash
cp .env.example .env
```

```
ANTHROPIC_API_KEY=여기에_클로드_API_키_입력
OPENAI_API_KEY=여기에_오픈AI_API_키_입력
```

### 3. 세법 데이터 추가

`data/laws/` 또는 `data/cases/` 폴더에 PDF 또는 TXT 파일을 넣습니다.

### 4. 벡터DB 구축

```bash
python -c "from src.indexer import build_index; build_index()"
```

### 5. 앱 실행

```bash
streamlit run app.py
```

---

## 개발 로드맵

- [x] 프로젝트 설계 및 환경 구성
- [ ] 데이터 로더 구현 (PDF/TXT 파싱)
- [ ] 벡터DB 인덱싱 구현
- [ ] RAG 파이프라인 구현
- [ ] Streamlit UI 구현
- [ ] 출처 명시 기능
- [ ] 대화 히스토리 기능

---

## 주의사항

- `.env` 파일에는 API 키가 포함되어 있으므로 절대 GitHub에 올리지 않습니다.
- `db/` 폴더는 용량이 크므로 git에서 제외합니다.
- `data/` 폴더의 세법 자료는 저작권에 유의합니다.
