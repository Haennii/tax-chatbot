"""
법제처 Open API에서 세법 조문 원문을 가져와 data/laws/ 에 저장합니다.
실행: python -m src.fetcher
"""

import requests
import xml.etree.ElementTree as ET
import json
from pathlib import Path
import config

# 가져올 세법 목록
TARGET_LAWS = [
    "법인세법", "소득세법", "부가가치세법", "조세특례제한법",
    "법인세법 시행령", "소득세법 시행령", "조세특례제한법 시행령",
]

SEARCH_URL = "https://www.law.go.kr/DRF/lawSearch.do"
LAW_URL = "https://www.law.go.kr/DRF/lawService.do"


def get_law_id(law_name: str) -> str:
    """법령 이름으로 법제처 법령 ID 조회"""
    params = {
        "OC": config.MOLEG_API_KEY,
        "target": "law",
        "type": "XML",
        "query": law_name,
        "display": 1,
        "page": 1,
    }
    r = requests.get(SEARCH_URL, params=params, timeout=10)
    r.raise_for_status()

    root = ET.fromstring(r.content)
    law = root.find(".//law")
    if law is None:
        raise ValueError(f"법령을 찾을 수 없습니다: {law_name}")

    law_id = law.findtext("법령ID")
    if not law_id:
        raise ValueError(f"법령 ID가 없습니다: {law_name}")
    return law_id


def fetch_articles(law_id: str) -> list[dict]:
    """법령 ID로 전체 조문 파싱"""
    params = {
        "OC": config.MOLEG_API_KEY,
        "target": "law",
        "type": "XML",
        "ID": law_id,
    }
    r = requests.get(LAW_URL, params=params, timeout=30)
    r.raise_for_status()

    root = ET.fromstring(r.content)
    law_name = root.findtext(".//법령명_한글", "")
    effective_date = root.findtext(".//시행일자", "")

    articles = []
    for unit in root.findall(".//조문단위"):
        num = unit.findtext("조문번호", "").strip()
        title = unit.findtext("조문제목", "").strip()
        body = unit.findtext("조문내용", "").strip()

        # 항(①②③...) → 호(1.2.3.) → 목(가.나.다.) 재귀 수집
        def collect_text(elem) -> str:
            parts = []
            body_text = elem.findtext("항내용") or elem.findtext("호내용") or elem.findtext("목내용") or ""
            if body_text.strip():
                parts.append(body_text.strip())
            for child_tag in ["호", "목"]:
                for child in elem.findall(child_tag):
                    child_text = collect_text(child)
                    if child_text:
                        parts.append(child_text)
            return "\n".join(parts)

        paragraphs = []
        for para in unit.findall("항"):
            para_text = collect_text(para)
            if para_text:
                paragraphs.append(para_text)

        full_text = body
        if paragraphs:
            full_text = (body + "\n" + "\n".join(paragraphs)).strip()

        if not full_text:
            continue

        articles.append({
            "law": law_name,
            "effective_date": effective_date,
            "article_num": num,
            "article_title": title,
            "text": full_text,
        })

    return articles


def save_articles(law_name: str, articles: list[dict]):
    """조문 목록을 JSON 파일로 저장"""
    Path(config.LAWS_DIR).mkdir(parents=True, exist_ok=True)
    path = Path(config.LAWS_DIR) / f"{law_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"  저장 완료: {path.name} ({len(articles)}개 조문)")


def fetch_all():
    """3개 세법 전체 갱신"""
    if not config.MOLEG_API_KEY:
        print("오류: MOLEG_API_KEY가 .env에 설정되지 않았습니다.")
        return

    for law_name in TARGET_LAWS:
        print(f"\n[{law_name}] 조회 중...")
        try:
            law_id = get_law_id(law_name)
            print(f"  법령 ID: {law_id}")
            articles = fetch_articles(law_id)
            save_articles(law_name, articles)
        except Exception as e:
            print(f"  오류: {e}")

    print("\n완료. 이제 python -m src.indexer 를 실행하세요.")


if __name__ == "__main__":
    fetch_all()
