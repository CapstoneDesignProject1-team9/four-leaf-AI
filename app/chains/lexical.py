"""BM25 키워드 검색(한글 글자 2-gram). 벡터 검색이 놓치는 공지 제목·과목명 일치를 보완한다."""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

K1, B = 1.2, 0.75


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text)


class Bm25Index:
    def __init__(self, texts: list[str]) -> None:
        # 형태소 분석기 없이 한국어 조사·어미 변화에 덜 민감하도록 글자 2-gram을 쓴다.
        self._vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(2, 2))
        tf = self._vectorizer.fit_transform([_normalize(t) for t in texts]).tocsc()
        self._tf = tf.astype(np.float32)
        length = np.asarray(tf.sum(axis=1)).ravel()
        df = np.diff(tf.indptr)
        self._idf = np.log(1 + (len(texts) - df + 0.5) / (df + 0.5))
        self._length_norm = K1 * (1 - B + B * length / max(length.mean(), 1))

    def search(self, query: str, n: int) -> list[int]:
        """점수가 높은 문서 번호를 최대 n개 돌려준다. 겹치는 글자가 없으면 빈 목록."""
        terms = self._vectorizer.transform([_normalize(query)]).indices
        if len(terms) == 0:
            return []
        tf = self._tf[:, terms].toarray()
        scores = (tf * (K1 + 1) / (tf + self._length_norm[:, None]) * self._idf[terms]).sum(axis=1)
        top = np.argsort(-scores)[:n]
        return [int(i) for i in top if scores[i] > 0]
