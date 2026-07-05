"""
data/laws/*.json 파일을 읽어 LangChain Document 객체로 변환합니다.
"""

import json
from pathlib import Path
from langchain.schema import Document
import config


def load_laws() -> list[Document]:
    """저장된 세법 JSON 파일을 Document 목록으로 변환"""
    docs = []
    law_dir = Path(config.LAWS_DIR)

    if not any(law_dir.glob("*.json")):
        print("세법 데이터가 없습니다. python -m src.fetcher 를 먼저 실행하세요.")
        return docs

    for path in sorted(law_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            articles = json.load(f)

        for a in articles:
            # 조문 제목 + 본문을 하나의 텍스트로 구성
            header = f"[{a['law']} 제{a['article_num']}조 {a['article_title']}]"
            content = f"{header}\n{a['text']}"

            docs.append(Document(
                page_content=content,
                metadata={
                    "law": a["law"],
                    "article_num": a["article_num"],
                    "article_title": a["article_title"],
                    "effective_date": a["effective_date"],
                },
            ))

    print(f"총 {len(docs)}개 조문 로드 완료")
    return docs
