"""
law.go.kr에서 부가가치세법·법인세법·소득세법 원문을 긁어와
data/laws/tax_context.txt 에 저장합니다.
"""

from playwright.sync_api import sync_playwright
from pathlib import Path
import time
import re

OUTPUT = Path("data/laws/tax_context.txt")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

LAWS = [
    ("부가세", "https://www.law.go.kr/법령/부가가치세법"),
    ("법인세", "https://www.law.go.kr/법령/법인세법"),
    ("소득세", "https://www.law.go.kr/법령/소득세법"),
]


def get_iframe_url(page, main_url: str) -> str:
    page.goto(main_url, wait_until="networkidle", timeout=60000)
    time.sleep(2)
    return page.eval_on_selector("#lawService", "el => el.src")


def scrape_law_text(page, iframe_url: str) -> str:
    page.goto(iframe_url, wait_until="networkidle", timeout=60000)
    time.sleep(4)
    text = page.locator("body").inner_text()
    return text.strip()


def clean_text(text: str) -> str:
    # 반복 공백·빈줄 정리
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def main():
    blocks = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_extra_http_headers({"Accept-Language": "ko-KR,ko;q=0.9"})

        for name, url in LAWS:
            print(f"\n[{name}] 시작...")
            try:
                iframe_url = get_iframe_url(page, url)
                print(f"  iframe URL 확보: {iframe_url}")

                text = scrape_law_text(page, iframe_url)
                text = clean_text(text)
                print(f"  조문 추출 완료: {len(text):,}자")

                blocks.append(f"== {name} ==\n\n{text}")
            except Exception as e:
                print(f"  오류: {e}")

        browser.close()

    OUTPUT.write_text("\n\n\n".join(blocks), encoding="utf-8")
    size_kb = OUTPUT.stat().st_size // 1024
    print(f"\n저장 완료 → {OUTPUT}  ({size_kb:,} KB)")


if __name__ == "__main__":
    main()
