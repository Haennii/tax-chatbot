"""
사용자 질문 → BM25 키워드 검색 → 율표 직접 파싱 + Ollama 호출 → 답변 반환
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

INTRO_PROMPT = PromptTemplate(
    input_variables=["article_title", "law", "article_num", "question"],
    template="""당신은 대한민국 세법 전문가입니다. 한국어로만 답변하세요.
질문: {question}
근거 조문: {law} 제{article_num}조 ({article_title})
위 조문을 근거로 한 문장으로 간단히 소개하세요. 율표 내용은 이미 아래에 별도로 표시되므로 율 숫자는 언급하지 마세요.
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


def clean_raw(text: str) -> str:
    text = re.sub(r'</?[^>]+>', ' ', text)
    text = re.sub(r'[┌┐└┘├┤┬┴┼─│]+', ' ', text)
    text = re.sub(r'[\t ]+', ' ', text)
    return text


def parse_rate_table(raw_text: str) -> str | None:
    """율표를 파싱해 카테고리별로 정리된 문자열 반환. 율표가 없으면 None."""
    text = clean_raw(raw_text)
    if '분의' not in text:
        return None

    # 한 줄로 합치기
    single = ' '.join(l.strip() for l in text.splitlines() if l.strip())

    # 메인 카테고리로 분할 (숫자. 으로 시작하는 부분)
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

        # 소항목 시작 위치 탐색 (공백 뒤 가/나/다 + 마침표)
        sub_start = re.search(r'(?:^|(?<=\s))([가나다라마바사아자차카타파하])\.', rest)

        if not sub_start:
            # 소항목 없이 바로 율이 있는 경우 (예: "제1호 및 제2호 외의 사업 102분의 2")
            rate_m = re.search(r'(\d+분의\s*\d+)', rest)
            if rate_m:
                desc = rest[:rate_m.start()].strip()
                result_parts.append(f'\n• {desc}: {rate_m.group(1)}')
            continue

        cat_name = rest[:sub_start.start()].strip()
        sub_text = rest[sub_start.start():]

        # 소항목 분할: 공백 뒤에 가/나/다 + 마침표
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

            # 괄호 제거한 after로 설명 보완
            after_no_paren = re.sub(r'\([^)]*\)', '', after).strip()

            if before:
                desc = (before + ' ' + after_no_paren).strip() if after_no_paren else before
            else:
                desc = after_no_paren

            # 가목/나목 참조 제거 → 자연스러운 표현으로
            desc = re.sub(
                r'[가나다라마바사아자차카타파하]목\s*및\s*[가나다라마바사아자차카타파하]목\s*외의\s*',
                '그 외 ', desc
            )
            desc = re.sub(r'[가나다라마바사아자차카타파하]목\s*외의\s*', '', desc)
            desc = desc.strip()

            # 율이 괄호 안에 있는 특례 표시
            special = re.search(r'\(([^)]*\d+분의\s*\d+[^)]*)\)', content)
            if special:
                result_parts.append(f'  - {desc}: {rate}  (※ {special.group(1).strip()})')
            else:
                result_parts.append(f'  - {desc}: {rate}')

    if not result_parts:
        return None

    return '\n'.join(result_parts).strip()


def clean_text(text: str) -> str:
    text = clean_raw(text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()


def load_all_articles():
    articles = []
    for path in Path(config.LAWS_DIR).glob("*.json"):
        with open(path, encoding="utf-8") as f:
            for a in json.load(f):
                a['_raw_text'] = a['text']
                a['text'] = clean_text(a['text'])
                articles.append(a)
    return articles


def tokenize(text: str) -> list[str]:
    return [token.form for token in kiwi.tokenize(text)]


MIN_RELEVANCE_SCORE = 5.0  # 이 점수 미만이면 관련 조문 없음으로 판단

LAW_NAME_MAP = {
    '소득세': '소득세법',
    '법인세': '법인세법',
    '부가세': '부가가치세법',
    '부가가치세': '부가가치세법',
    '조세특례': '조세특례제한법',
    '세액공제': '조세특례제한법',
}


def bm25_search(query: str, articles: list, k: int = 5) -> tuple[list, float]:
    """(검색결과, 최고점수) 반환"""
    corpus = [a["text"] for a in articles]
    tokenized = [tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(tokenize(query))

    query_tokens = set(tokenize(query))

    # 질문에 법령명이 포함되면 해당 법령 조문 우선
    preferred_law = None
    for keyword, law_name in LAW_NAME_MAP.items():
        if keyword in query:
            preferred_law = law_name
            break

    # "소득세율", "법인세율" 같은 합성어 → "세율" 토큰 명시 추가
    import re as _re
    if _re.search(r'[가-힣]+세율$', query):
        query_tokens.add('세율')

    boosted = []
    for i, score in enumerate(scores):
        title_tokens = set(tokenize(articles[i]["article_title"]))
        bonus = len(query_tokens & title_tokens) * 3.0
        # 법령명 일치 보너스
        if preferred_law and articles[i]["law"] == preferred_law:
            bonus += 5.0
        # "소득세율" 같은 합성 세율 질문 → 해당 법령의 "세율" 제목 조문 강력 우선
        if preferred_law and '세율' in query_tokens and articles[i]["law"] == preferred_law:
            if '세율' in articles[i]["article_title"]:
                bonus += 15.0
            # "세율" 단독 제목인 경우: 조번호 작을수록 일반 세율 → 우선
            if articles[i]["article_title"].strip() == '세율':
                try:
                    bonus += 500 / (int(articles[i]["article_num"]) + 1)
                except ValueError:
                    bonus += 5.0
        boosted.append((i, score + bonus))

    top_indices = sorted(boosted, key=lambda x: x[1], reverse=True)[:k]
    top_score = boosted[top_indices[0][0]][1] if top_indices else 0.0
    return [articles[i] for i, _ in top_indices], top_score


NO_RESULT_MSG = (
    "죄송합니다. 현재 데이터에는 해당 질문과 관련된 조문이 없습니다.\n\n"
    "현재 수록된 법령: 법인세법, 소득세법, 부가가치세법, 조세특례제한법"
)


def _prepare(question: str):
    """검색 + 율표 파싱. (top_articles, rate_table, sources, no_result) 반환."""
    articles = load_all_articles()
    top_articles, top_score = bm25_search(question, articles, k=config.TOP_K)

    if top_score < MIN_RELEVANCE_SCORE:
        return [], None, [], True

    q_tokens = set(tokenize(question))
    top_title_tokens = set(tokenize(top_articles[0]["article_title"]))
    top_articles = top_articles[:1] if len(q_tokens & top_title_tokens) >= 2 else top_articles[:2]

    top = top_articles[0]
    has_table = any(c in top['_raw_text'] for c in '┌┐└┘├┤┬┴┼─│')
    rate_table = parse_rate_table(top['_raw_text']) if has_table else None

    sources = [
        {"law": a["law"], "article_num": a["article_num"],
         "article_title": a["article_title"], "effective_date": a["effective_date"]}
        for a in top_articles
    ]
    return top_articles, rate_table, sources, False


def ask(question: str) -> dict:
    """동기 호출용 (테스트 등)"""
    top_articles, rate_table, sources, no_result = _prepare(question)
    if no_result:
        return {"answer": NO_RESULT_MSG, "sources": []}

    llm = ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0)
    top = top_articles[0]

    if rate_table:
        intro = llm.invoke(INTRO_PROMPT.format(
            article_title=top['article_title'], law=top['law'],
            article_num=top['article_num'], question=question,
        )).content.strip()
        answer = f"{intro}\n\n{rate_table}\n\n[근거: {top['law']} 제{top['article_num']}조 {top['article_title']}]"
    else:
        context = "\n\n".join(
            f"[{a['law']} 제{a['article_num']}조 {a['article_title']}]\n{a['text'][:800]}"
            for a in top_articles
        )
        answer = llm.invoke(GENERAL_PROMPT.format(context=context, question=question)).content

    return {"answer": answer, "sources": sources}


def ask_stream(question: str):
    """스트리밍 호출용. (chunk_text, sources_or_None) 를 yield."""
    top_articles, rate_table, sources, no_result = _prepare(question)
    if no_result:
        yield NO_RESULT_MSG, sources
        return

    llm = ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0)
    top = top_articles[0]

    if rate_table:
        # 율표: 소개문 스트리밍 후 파싱된 표 즉시 출력
        intro_chunks = []
        for chunk in llm.stream(INTRO_PROMPT.format(
            article_title=top['article_title'], law=top['law'],
            article_num=top['article_num'], question=question,
        )):
            intro_chunks.append(chunk.content)
            yield chunk.content, None
        suffix = f"\n\n{rate_table}\n\n[근거: {top['law']} 제{top['article_num']}조 {top['article_title']}]"
        yield suffix, sources
    else:
        context = "\n\n".join(
            f"[{a['law']} 제{a['article_num']}조 {a['article_title']}]\n{a['text'][:800]}"
            for a in top_articles
        )
        for chunk in llm.stream(GENERAL_PROMPT.format(context=context, question=question)):
            yield chunk.content, None
        yield "", sources
