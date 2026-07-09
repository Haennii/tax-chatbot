"""
검색 정확도 자동 테스트
실행: python -m tests.test_search
"""
import sys
sys.path.insert(0, '.')
from src.chatbot import load_all_articles, search_articles

# (질문, 기대 법령명, 기대 조번호)
TEST_CASES = [
    ("법인세율",                     "법인세법",          "55"),
    ("법인세 세율",                  "법인세법",          "55"),
    ("소득세율",                     "소득세법",          "55"),
    ("소득세 세율",                  "소득세법",          "55"),
    ("양도소득세율",                 "소득세법",          "104"),
    ("최저한세율",                   "조세특례제한법",    "132"),
    ("법인세 최저한세율",            "조세특례제한법",    "132"),
    ("의제매입세액 공제율",          "부가가치세법",      "42"),
    ("부가세 신고기한",              "부가가치세법",      "49"),
    ("연구인력개발세액공제율",       "조세특례제한법",    "10"),
    ("임원 퇴직급여 한도",          "법인세법 시행령",   "44"),
    ("임원 퇴직급여 한도 산식",     "법인세법 시행령",   "44"),
    ("접대비 손금한도",             "법인세법",          "25"),
    ("근로소득공제",                "소득세법",          "47"),
    ("인적공제",                    "소득세법",          "50"),
]


def run():
    print("조문 로딩 중...")
    articles = load_all_articles()
    print(f"총 {len(articles)}개 조문\n")

    passed = 0
    failed = []

    for question, exp_law, exp_num in TEST_CASES:
        results, by_concept = search_articles(question, articles)
        method = "개념매핑" if by_concept else "BM25"

        if not results:
            failed.append((question, exp_law, exp_num, "결과없음", ""))
            print(f"  FAIL [{method}] '{question}' → 결과 없음")
            continue

        top = results[0]
        ok = (top["law"] == exp_law and top["article_num"] == exp_num)

        if ok:
            passed += 1
            print(f"  OK   [{method}] '{question}' → {top['law']} 제{top['article_num']}조")
        else:
            failed.append((question, exp_law, exp_num, top["law"], top["article_num"]))
            print(f"  FAIL [{method}] '{question}'")
            print(f"       기대: {exp_law} 제{exp_num}조")
            print(f"       실제: {top['law']} 제{top['article_num']}조 {top['article_title']}")

    total = len(TEST_CASES)
    print(f"\n결과: {passed}/{total} 통과 ({passed/total*100:.0f}%)")
    return passed == total


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
