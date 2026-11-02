"""루트 conftest.py — GPU/HF 인증이 필요한 스크립트와 Playwright 테스트를 CI에서 제외."""

collect_ignore = [
    # 실제 Llama 3.1 8B 모델을 HuggingFace에서 다운로드해야 하는 어댑터 비교 스크립트.
    # CI 환경에는 GPU도 HF 토큰도 없으므로 수집 단계에서 제외.
    "test_adapters.py",
]

collect_ignore_glob = [
    # Playwright 브라우저가 필요한 에브리타임 크롤러 테스트. CI에서 skip.
    "tests/everytime/*",
]
