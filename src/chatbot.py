"""
사용자 질문 → 개념 직접 매핑(우선) or BM25 검색(폴백) → 율표 파싱 or LLM → 답변
"""

import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
import config

kiwi = Kiwi()

# ──────────────────────────────────────────────
# 프롬프트
# ──────────────────────────────────────────────
INTRO_PROMPT = PromptTemplate(
    input_variables=["article_title", "law", "article_num", "question"],
    template="""당신은 대한민국 세법 전문가입니다. 한국어로만 답변하세요.
질문: {question}
근거 조문: {law} 제{article_num}조 ({article_title})
위 조문을 근거로 한 문장으로 간단히 소개하세요. 율표 내용은 이미 아래에 별도로 표시됩니다.
[답변]""",
)

GENERAL_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""당신은 대한민국 세법 전문가입니다. 반드시 한국어로만 답변하세요.

아래 조문을 근거로 질문에 답하세요. 조문에 없는 내용은 추측하지 마세요.
답변 마지막에 근거 조문(법령명, 조번호)을 명시하세요.

[세법 조문]
{context}

[질문]
{question}

[한국어 답변]""",
)

# ──────────────────────────────────────────────
# 개념 직접 매핑 (법령명, 조번호)
# 자주 묻는 질문 / BM25로 잘못 찾는 케이스를 직접 지정
# ──────────────────────────────────────────────
CONCEPT_MAP: list[tuple[list[str], str, str]] = [
    # (매칭 키워드 목록, 법령명, 조번호)
    # ── 세율 (구체적인 것 먼저) ──
    (["양도소득세율", "양도세율", "양도소득세 세율"], "소득세법", "104"),
    (["법인세율", "법인세 세율", "법인세 과세표준"], "법인세법", "55"),
    (["소득세율", "소득세 세율", "소득세 과세표준"], "소득세법", "55"),
    (["최저한세율", "최저한세", "법인세 최저한세"], "조세특례제한법", "132"),
    # ── 부가가치세 ──
    (["의제매입세액", "의제매입세액 공제율"], "부가가치세법", "42"),
    (["부가가치세 신고", "부가세 신고", "예정신고", "확정신고 납부"], "부가가치세법", "49"),
    (["영세율"], "부가가치세법", "21"),
    # ── 손금/필요경비 ──
    (["임원 퇴직급여 한도", "임원퇴직급여한도", "임원 퇴직급여 산식", "임원 퇴직급여 한도 산식"], "법인세법 시행령", "44"),
    (["접대비 한도", "접대비"], "법인세법", "25"),
    (["감가상각"], "법인세법", "23"),
    # ── 세액공제 ──
    (["연구인력개발", "연구개발세액공제", "연구비 세액공제"], "조세특례제한법", "10"),
    (["중소기업 특별세액감면"], "조세특례제한법", "7"),
    # ── 소득세 ──
    (["근로소득공제"], "소득세법", "47"),
    (["인적공제", "기본공제", "추가공제"], "소득세법", "50"),
    (["연금보험료공제"], "소득세법", "51"),
    (["특별소득공제"], "소득세법", "52"),
    (["근로소득세액공제"], "소득세법", "59"),
    (["퇴직소득세"], "소득세법", "22"),
    # ── 법인세 신고/납부 ──
    (["법인세 신고", "법인세 신고기한", "법인세 납세"], "법인세법", "60"),
]


def find_by_concept(question: str, articles: list[dict]) -> list[dict]:
    """개념 매핑으로 조문 직접 반환. 없으면 빈 리스트."""
    q = question.replace(" ", "")  # 공백 제거 후 매칭
    for keywords, law_name, article_num in CONCEPT_MAP:
        for kw in keywords:
            if kw.replace(" ", "") in q:
                matched = [
                    a for a in articles
                    if a["law"] == law_name and a["article_num"] == article_num
                ]
                if matched:
                    return matched
    return []


# ──────────────────────────────────────────────
# 텍스트 정제
# ──────────────────────────────────────────────
def clean_raw(text: str) -> str:
    text = re.sub(r'</?[^>]+>', ' ', text)
    text = re.sub(r'[┌┐└┘├┤┬┴┼─│]+', ' ', text)
    text = re.sub(r'[\t ]+', ' ', text)
    return text


def clean_text(text: str) -> str:
    text = clean_raw(text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()


# ──────────────────────────────────────────────
# 율표 직접 파싱 (박스 문자 테이블)
# ──────────────────────────────────────────────
def parse_rate_table(raw_text: str) -> str | None:
    text = clean_raw(raw_text)
    if '분의' not in text:
        return None

    single = ' '.join(l.strip() for l in text.splitlines() if l.strip())
    sections = re.split(r'(?<!\w)(?=\d+\.\s)', single)
    result_parts = []

    for section in sections:
        section = section.strip()
        if '분의' not in section:
            continue

        num_match = re.match(r'^(\d+)\.\s*(.+)', section)
        if not num_match:
            continue

        rest = num_match.group(2).strip()
        sub_start = re.search(r'(?:^|(?<=\s))([가나다라마바사아자차카타파하])\.', rest)

        if not sub_start:
            rate_m = re.search(r'(\d+분의\s*\d+)', rest)
            if rate_m:
                desc = rest[:rate_m.start()].strip()
                result_parts.append(f'\n• {desc}: {rate_m.group(1)}')
            continue

        cat_name = rest[:sub_start.start()].strip()
        sub_text = rest[sub_start.start():]
        sub_parts = re.split(r'(?<=\s)(?=[가나다라마바사아자차카타파하]\.)', sub_text)

        result_parts.append(f'\n■ {cat_name}')

        for sub_part in sub_parts:
            sub_m = re.match(r'^([가나다라마바사아자차카타파하])\.\s*(.+)', sub_part)
            if not sub_m:
                continue
            content = sub_m.group(2).strip()
            rate_m = re.search(r'(\d+분의\s*\d+)', content)
            if not rate_m:
                continue

            before = content[:rate_m.start()].strip()
            after = content[rate_m.end():].strip()
            rate = rate_m.group(1)
            after_no_paren = re.sub(r'\([^)]*\)', '', after).strip()

            if before:
                desc = (before + ' ' + after_no_paren).strip() if after_no_paren else before
            else:
                desc = after_no_paren

            desc = re.sub(r'[가나다라마바사아자차카타파하]목\s*및\s*[가나다라마바사아자차카타파하]목\s*외의\s*', '그 외 ', desc)
            desc = re.sub(r'[가나다라마바사아자차카타파하]목\s*외의\s*', '', desc)
            desc = desc.strip()

            special = re.search(r'\(([^)]*\d+분의\s*\d+[^)]*)\)', content)
            if special:
                result_parts.append(f'  - {desc}: {rate}  (※ {special.group(1).strip()})')
            else:
                result_parts.append(f'  - {desc}: {rate}')

    if not result_parts:
        return None
    return '\n'.join(result_parts).strip()


# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────
def load_all_articles() -> list[dict]:
    articles = []
    for path in Path(config.LAWS_DIR).glob("*.json"):
        with open(path, encoding="utf-8") as f:
            for a in json.load(f):
                a['_raw_text'] = a['text']
                a['text'] = clean_text(a['text'])
                articles.append(a)
    return articles


# ──────────────────────────────────────────────
# BM25 검색 (개념 매핑 실패 시 폴백)
# ──────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    return [token.form for token in kiwi.tokenize(text)]


MIN_RELEVANCE_SCORE = 5.0


def bm25_search(query: str, articles: list, k: int = 5) -> tuple[list, float]:
    corpus = [a["text"] for a in articles]
    tokenized = [tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(tokenize(query))

    query_tokens = set(tokenize(query))
    boosted = []
    for i, score in enumerate(scores):
        title_tokens = set(tokenize(articles[i]["article_title"]))
        bonus = len(query_tokens & title_tokens) * 3.0
        boosted.append((i, score + bonus))

    top_indices = sorted(boosted, key=lambda x: x[1], reverse=True)[:k]
    top_score = boosted[top_indices[0][0]][1] if top_indices else 0.0
    return [articles[i] for i, _ in top_indices], top_score


# ──────────────────────────────────────────────
# 검색 진입점 (개념 매핑 → BM25 폴백)
# ──────────────────────────────────────────────
NO_RESULT_MSG = (
    "죄송합니다. 현재 데이터에는 해당 질문과 관련된 조문이 없습니다.\n\n"
    "수록 법령: 법인세법/시행령, 소득세법/시행령, 부가가치세법, 조세특례제한법/시행령"
)


def search_articles(question: str, articles: list) -> tuple[list, bool]:
    """(조문 목록, 개념매핑 여부) 반환"""
    # 1단계: 개념 직접 매핑
    matched = find_by_concept(question, articles)
    if matched:
        return matched, True

    # 2단계: BM25 폴백
    results, top_score = bm25_search(question, articles, k=config.TOP_K)
    if top_score < MIN_RELEVANCE_SCORE:
        return [], False

    q_tokens = set(tokenize(question))
    top_title_tokens = set(tokenize(results[0]["article_title"]))
    results = results[:1] if len(q_tokens & top_title_tokens) >= 2 else results[:2]
    return results, False


# ──────────────────────────────────────────────
# 답변 생성
# ──────────────────────────────────────────────
def _build_answer_sync(top_articles: list, question: str, llm) -> str:
    top = top_articles[0]
    has_table = any(c in top['_raw_text'] for c in '┌┐└┘├┤┬┴┼─│')
    rate_table = parse_rate_table(top['_raw_text']) if has_table else None

    if rate_table:
        intro = llm.invoke(INTRO_PROMPT.format(
            article_title=top['article_title'], law=top['law'],
            article_num=top['article_num'], question=question,
        )).content.strip()
        return f"{intro}\n\n{rate_table}\n\n[근거: {top['law']} 제{top['article_num']}조 {top['article_title']}]"
    else:
        context = "\n\n".join(
            f"[{a['law']} 제{a['article_num']}조 {a['article_title']}]\n{a['text'][:1000]}"
            for a in top_articles
        )
        return llm.invoke(GENERAL_PROMPT.format(context=context, question=question)).content


def ask(question: str) -> dict:
    articles = load_all_articles()
    top_articles, _ = search_articles(question, articles)

    if not top_articles:
        return {"answer": NO_RESULT_MSG, "sources": []}

    llm = ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0)
    answer = _build_answer_sync(top_articles, question, llm)

    sources = [
        {"law": a["law"], "article_num": a["article_num"],
         "article_title": a["article_title"], "effective_date": a["effective_date"]}
        for a in top_articles
    ]
    return {"answer": answer, "sources": sources}


def ask_stream(question: str):
    """스트리밍 호출용. (chunk_text, sources_or_None) yield."""
    articles = load_all_articles()
    top_articles, _ = search_articles(question, articles)

    if not top_articles:
        yield NO_RESULT_MSG, []
        return

    llm = ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0)
    top = top_articles[0]
    has_table = any(c in top['_raw_text'] for c in '┌┐└┘├┤┬┴┼─│')
    rate_table = parse_rate_table(top['_raw_text']) if has_table else None

    sources = [
        {"law": a["law"], "article_num": a["article_num"],
         "article_title": a["article_title"], "effective_date": a["effective_date"]}
        for a in top_articles
    ]

    if rate_table:
        for chunk in llm.stream(INTRO_PROMPT.format(
            article_title=top['article_title'], law=top['law'],
            article_num=top['article_num'], question=question,
        )):
            yield chunk.content, None
        suffix = f"\n\n{rate_table}\n\n[근거: {top['law']} 제{top['article_num']}조 {top['article_title']}]"
        yield suffix, sources
    else:
        context = "\n\n".join(
            f"[{a['law']} 제{a['article_num']}조 {a['article_title']}]\n{a['text'][:1000]}"
            for a in top_articles
        )
        for chunk in llm.stream(GENERAL_PROMPT.format(context=context, question=question)):
            yield chunk.content, None
        yield "", sources
