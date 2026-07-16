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
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import config

kiwi = Kiwi()

# ──────────────────────────────────────────────
# 프롬프트
# ──────────────────────────────────────────────
SYSTEM_ROLE = (
    "당신은 대한민국 세법 전문가로서 20년 이상의 세무사 경력을 보유하고 있습니다. "
    "법인세법, 소득세법, 부가가치세법, 조세특례제한법 및 각 시행령에 정통하며, "
    "납세자와 세무 실무자가 이해하기 쉽도록 정확하고 명확하게 답변합니다. "
    "반드시 한국어로만 답변하고, 제시된 조문 외의 내용은 추측하지 않습니다."
)

INTRO_PROMPT = PromptTemplate(
    input_variables=["article_title", "law", "article_num", "question"],
    template="""{system_role}

질문: {{question}}
근거 조문: {{law}} 제{{article_num}}조 ({{article_title}})
세법 전문가로서 위 조문을 근거로 한 문장으로 핵심만 설명하세요. 율표 내용은 별도로 표시됩니다.
[답변]""".replace("{system_role}", SYSTEM_ROLE),
)

GENERAL_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""{system_role}

아래 [세법 조문]만을 근거로 질문에 답하세요.
- 날짜, 금액, 비율 등 모든 수치는 반드시 아래 조문에서 읽은 값만 사용하세요.
- 조문의 내용이 당신이 알고 있는 내용과 다르더라도 조문을 우선합니다.
- 조문에 없는 내용은 절대 추측하거나 추가하지 마세요.
먼저 조문에서 질문과 관련된 내용을 찾은 뒤, 핵심 내용을 조목조목 설명하세요. 답변 마지막에 근거 조문(법령명, 조번호)을 명시하세요.

[답변 예시]
질문: 법인세 신고기한은 언제인가요?
답변: 내국법인은 각 사업연도 종료일이 속하는 달의 말일부터 3개월 이내에 법인세 과세표준과 세액을 신고해야 합니다.
근거: 법인세법 제60조

질문: 접대비 손금한도는 얼마인가요?
답변: 접대비 손금 한도는 기본한도(중소기업 3,600만 원, 그 외 1,200만 원)에 수입금액 기준 한도를 합산한 금액입니다.
근거: 법인세법 제25조

[세법 조문]
{{context}}

[질문]
{{question}}

[세법 전문가 답변]""".replace("{system_role}", SYSTEM_ROLE),
)

# ──────────────────────────────────────────────
# 개념 직접 매핑 (법령명, 조번호)
# 자주 묻는 질문 / BM25로 잘못 찾는 케이스를 직접 지정
# ──────────────────────────────────────────────
CONCEPT_MAP: list[tuple[list[str], str, str]] = [
    # (매칭 키워드 목록, 법령명, 조번호)
    # ── 세율 (구체적인 것 먼저) ──
    (["양도소득세율", "양도세율", "양도소득세 세율"], "소득세법", "104"),
    (["양도소득세 신고", "양도세 신고", "양도소득 예정신고", "양도세 신고기한"], "소득세법", "105"),
    (["양도소득세 확정신고", "양도소득 확정신고"], "소득세법", "110"),
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
    # ── 소득세 신고/납부 (구체적인 것 먼저) ──
    (["성실신고", "성실신고확인", "성실신고대상", "성실신고 신고기한"], "소득세법", "70", "성실신고확인서 제출"),
    (["소득세 신고기한", "종합소득세 신고", "종합소득 신고기한"], "소득세법", "70", "종합소득과세표준 확정신고"),
    # ── 특별세액공제 (59조에 여러 조문이 혼재 → 제목으로 구분) ──
    (["교육비 세액공제", "교육비공제", "교육비"], "소득세법", "59", "특별세액공제"),
    (["의료비 세액공제", "의료비공제", "의료비"], "소득세법", "59", "특별세액공제"),
]


def find_by_concept(question: str, articles: list[dict]) -> list[dict]:
    """개념 매핑으로 조문 직접 반환. 없으면 빈 리스트."""
    q = question.replace(" ", "")
    for entry in CONCEPT_MAP:
        keywords, law_name, article_num = entry[0], entry[1], entry[2]
        title_filter = entry[3] if len(entry) == 4 else None
        for kw in keywords:
            if kw.replace(" ", "") in q:
                matched = [
                    a for a in articles
                    if a["law"] == law_name and a["article_num"] == article_num
                    and (title_filter is None or a["article_title"] == title_filter)
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


def parse_deadline(raw_text: str) -> str | None:
    """신고기한 관련 조문에서 기한 문장을 직접 추출."""
    text = clean_raw(raw_text)
    patterns = [
        r'[^\n]*(?:신고|제출|납부)[^\n]*(?:\d+월\s*\d+일까지|이내)[^\n]*',
        r'[^\n]*\d+월\s*\d+일부터\s*\d+월\s*\d+일까지[^\n]*',
    ]
    found = []
    for pat in patterns:
        for m in re.finditer(pat, text):
            line = m.group().strip()
            if len(line) > 10:
                found.append(line)
    if not found:
        return None
    seen = []
    for f in found:
        if f not in seen:
            seen.append(f)
    return '\n'.join(f'• {s}' for s in seen[:5])


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
# 동의어/약어 확장 테이블
# ──────────────────────────────────────────────
SYNONYMS: dict[str, str] = {
    "양도세": "양도소득세",
    "부가세": "부가가치세",
    "법세": "법인세",
    "소세": "소득세",
    "신고기한": "예정신고 확정신고",
    "납부기한": "신고 납부",
    "공제율": "세액공제 공제",
    "한도액": "한도",
    "손금한도": "손금 한도",
    "세율표": "세율 과세표준",
    "퇴직금": "퇴직급여",
    "연구개발": "연구인력개발",
    "R&D": "연구인력개발",
    "의료비": "의료비 세액공제",
    "교육비": "교육비 세액공제",
}


def expand_query(query: str) -> str:
    """약어·동의어를 원어로 치환해 BM25 매칭률 향상."""
    for short, full in SYNONYMS.items():
        query = query.replace(short, full)
    return query


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
    # 1단계: 개념 직접 매핑 (동의어 확장 후 시도)
    matched = find_by_concept(question, articles)
    if matched:
        return matched, True

    # 2단계: BM25 폴백 (동의어 확장된 쿼리 사용)
    expanded = expand_query(question)
    results, top_score = bm25_search(expanded, articles, k=config.TOP_K)
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
            f"[{a['law']} 제{a['article_num']}조 {a['article_title']}]\n{a['text'][:2000]}"
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


def _build_messages(context: str, question: str, history: list[dict]) -> list:
    """대화 히스토리 + 현재 질문을 LangChain 메시지 목록으로 변환."""
    msgs = [SystemMessage(content=(
        SYSTEM_ROLE + " 답변 마지막에 근거 조문(법령명, 조번호)을 명시하세요."
    ))]
    for m in history:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            msgs.append(AIMessage(content=m["content"]))
    msgs.append(HumanMessage(content=f"[세법 조문]\n{context}\n\n[질문]\n{question}"))
    return msgs


def _expand_with_history(question: str, history: list[dict]) -> str:
    """짧거나 모호한 후속 질문에 직전 사용자 질문을 붙여 검색 정확도 향상."""
    if not history or len(question.replace(" ", "")) >= 15:
        return question
    prev = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
    return f"{prev} {question}" if prev else question


def ask_stream(question: str, history: list[dict] | None = None):
    """스트리밍 호출용. (chunk_text, sources_or_None) yield."""
    if history is None:
        history = []

    articles = load_all_articles()
    top_articles, _ = search_articles(question, articles)

    # 후속 질문으로 검색 실패 시 이전 질문과 합쳐서 재검색
    if not top_articles and history:
        expanded = _expand_with_history(question, history)
        if expanded != question:
            top_articles, _ = search_articles(expanded, articles)

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

    deadline = parse_deadline(top['_raw_text']) if any(kw in question for kw in ['신고기한', '납부기한', '제출기한', '신고기간', '신고']) else None

    if rate_table:
        for chunk in llm.stream(INTRO_PROMPT.format(
            article_title=top['article_title'], law=top['law'],
            article_num=top['article_num'], question=question,
        )):
            yield chunk.content, None
        suffix = f"\n\n{rate_table}\n\n[근거: {top['law']} 제{top['article_num']}조 {top['article_title']}]"
        yield suffix, sources
    elif deadline:
        ref = f"[근거: {top['law']} 제{top['article_num']}조 {top['article_title']}]"
        yield f"{deadline}\n\n{ref}", sources
    else:
        def build_context(article, q):
            text = article['text']
            best, best_score = '', 0
            for line in text.split('\n'):
                score = sum(1 for w in ['기한', '까지', '이내', '월', '일', '분의', '%'] if w in line)
                if score > best_score:
                    best_score, best = score, line.strip()
            header = f"[핵심 조항] {best}\n\n" if best_score >= 2 else ""
            return f"[{article['law']} 제{article['article_num']}조 {article['article_title']}]\n{header}{text[:2000]}"

        context = "\n\n".join(build_context(a, question) for a in top_articles)
        messages = _build_messages(context, question, history)
        for chunk in llm.stream(messages):
            yield chunk.content, None
        yield "", sources
