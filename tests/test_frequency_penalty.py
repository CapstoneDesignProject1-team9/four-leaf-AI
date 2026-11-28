import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from eval.generate_answers import frequency_penalty_processor  # noqa: E402


def test_penalizes_only_generated_tokens_by_count():
    # 프롬프트 [1, 2], 생성 [3, 3, 2] → 3은 2번, 2는 1번 (프롬프트의 2는 세지 않음)
    input_ids = torch.tensor([[1, 2, 3, 3, 2]])
    scores = torch.zeros(1, 5)
    out = frequency_penalty_processor(2, 0.5)(input_ids, scores)
    assert out.tolist() == [[0.0, 0.0, -0.5, -1.0, 0.0]]


def test_no_generated_tokens_leaves_scores():
    scores = torch.zeros(1, 5)
    out = frequency_penalty_processor(2, 0.5)(torch.tensor([[1, 2]]), scores)
    assert out.tolist() == [[0.0] * 5]
