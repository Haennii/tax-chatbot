# 세법 챗봇 (Tax Law Chatbot)

법제처 Open API로 최신 세법 조문을 자동으로 가져와 RAG(검색 증강 생성) 방식으로 답변하는 챗봇입니다.
법인세법 · 소득세법 · 부가가치세법을 커버하며, 답변 시 근거 조문을 함께 표시합니다.

---

## 작동 원리

```
사용자 질문
    ↓
ChromaDB에서 관련 세법 조문 검색
    ↓
Claude에게 "이 조문만 근거로 답변해" 지시
    ↓
근거 조문이 명시된 신뢰도 높은 답변
```

할루시네이션 방지: LLM이 자체 지식으로 답변하지 않고, 실제 조문 원문을 검색한 뒤 그 내용만을 근거로 답변합니다.

---

## 기술 스택

| 역할 | 기술 |
|---|---|
| UI | Streamlit |
| LLM | Claude API (Anthropic) |
| 벡터DB | ChromaDB |
| RAG 프레임워크 | LangChain |
| 임베딩 | OpenAI text-embedding-3-small |
| 세법 데이터 | 법제처 Open API |

---

## 프로젝트 구조

```
tax-chatbot/
│
├── app.py              # Streamlit 앱 실행 진입점
├── config.py           # 모델명, 경로, 파라미터 설정값
├── .env                # API 키 (git 제외)
├── .env.example        # API 키 템플릿 (git 포함)
├── requirements.txt    # 패키지 목록
│
├── data/
│   └── laws/           # 법제처 API로 받아온 조문 JSON 저장
│
├── db/                 # ChromaDB 벡터 저장소 (자동 생성, git 제외)
│
└── src/
    ├── fetcher.py      # 법제처 API → 조문 다운로드 → data/laws/ 저장
    ├── loader.py       # data/laws/ 파일 읽기 → Document 변환
    ├── indexer.py      # Document → 청크 분할 → 벡터DB 저장
    └── chatbot.py      # 질문 → 조문 검색 → Claude 호출 → 답변 반환
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

| 키 | 발급처 |
|---|---|
| `ANTHROPIC_API_KEY` | https://console.anthropic.com |
| `OPENAI_API_KEY` | https://platform.openai.com |
| `MOLEG_API_KEY` | https://www.law.go.kr (회원가입 후 아이디 사용) |

### 3. 세법 조문 다운로드 (법제처 API)

```bash
python -m src.fetcher
```

법인세법, 소득세법, 부가가치세법 조문을 `data/laws/`에 저장합니다.  
개정이 있을 때마다 이 명령어를 다시 실행하면 최신 조문으로 갱신됩니다.

### 4. 벡터DB 구축

```bash
python -m src.indexer
```

### 5. 앱 실행

```bash
streamlit run app.py
```

---

## 개발 로드맵

- [x] 프로젝트 설계 및 환경 구성
- [x] 법제처 Open API 연동 (조문 자동 갱신)
- [x] 데이터 로더 구현
- [x] 벡터DB 인덱싱 구현
- [x] RAG 파이프라인 구현
- [x] Streamlit UI 구현
- [x] 출처 명시 기능
- [x] 대화 히스토리 기능

---

## 주의사항

- `.env` 파일에는 API 키가 포함되어 있으므로 절대 GitHub에 올리지 않습니다.
- `db/` 폴더는 용량이 크므로 git에서 제외합니다.
- 법제처 Open API는 무료이며, 회원가입 후 로그인 아이디를 `MOLEG_API_KEY`로 사용합니다.
