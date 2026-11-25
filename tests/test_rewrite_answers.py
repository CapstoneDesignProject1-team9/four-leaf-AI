from train.rewrite_answers import problems

ROW = {
    "prompt": [
        {
            "role": "user",
            "content": "[공지 1]\n제목: 수강신청 안내\nURL: https://a.kr/1\n9월 1일 시작",
        }
    ],
    "completion": [{"role": "assistant", "content": "수강신청은 9월 1일에 시작합니다."}],
}


def test_passes_when_facts_and_source_are_kept():
    text = "수강신청은 9월 1일에 시작해요.\n\n참고: 수강신청 안내 (https://a.kr/1)"
    assert problems(text, ROW) == []


def test_flags_new_numbers_urls_and_missing_source():
    issues = problems("수강신청은 9월 3일에 시작해요. https://b.kr", ROW)
    assert any("새 숫자" in i for i in issues)
    assert any("새 URL" in i for i in issues)
    assert "출처 줄 없음" in issues
