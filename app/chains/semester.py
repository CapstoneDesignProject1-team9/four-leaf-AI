"""질문·공지 제목에서 학기를 읽는다. 공지 메타데이터에 게시일이 없어 제목이 유일한 단서다."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

SEASON_NAMES = {"1": "1학기", "여름": "여름계절학기", "2": "2학기", "겨울": "겨울계절학기"}

_YEAR = r"(?<!\d)(20\d{2})(?!\d)"
_SEP = r"\s*(?:학년도|년도|학년|년)?\s*[-.]?\s*(?:제\s*)?"
_FULL = re.compile(_YEAR + _SEP + r"([12])\s*학기")
_FULL_SEASONAL = re.compile(_YEAR + _SEP + r"(여름|하계|겨울|동계)\s*(?:계절|학기)")
_SEASON_ONLY = re.compile(r"(?<!\d)([12])\s*학기")
_SEASONAL_ONLY = re.compile(r"(여름|하계|겨울|동계)\s*(?:계절|학기)")
_ALIASES = {"하계": "여름", "동계": "겨울"}
_YEAR_ONLY = re.compile(_YEAR)
_RELATIVE = {
    0: re.compile(r"이번\s*학기|금\s*학기|현재\s*학기"),
    1: re.compile(r"다음\s*학기"),
    -1: re.compile(r"(?:지난|저번)\s*학기"),
}


@dataclass(frozen=True)
class Term:
    year: int
    season: str | None = None  # None이면 연도만 안다.

    def label(self) -> str:
        if self.season is None:
            return f"{self.year}학년도"
        return f"{self.year}학년도 {SEASON_NAMES[self.season]}"

    def shift(self, n: int) -> Term:
        """정규 학기(1·2) 기준으로 n학기 앞뒤."""
        index = self.year * 2 + (0 if self.season in ("1", "여름") else 1) + n
        return Term(index // 2, "1" if index % 2 == 0 else "2")

    def matches(self, other: Term) -> bool:
        if self.year != other.year:
            return False
        return self.season is None or other.season is None or self.season == other.season


def _season(text: str) -> str:
    return _ALIASES.get(text, text)


def current_term(today: date | None = None) -> Term:
    """1~6월은 1학기, 7~12월은 2학기. 방학 중에는 다가오는 학기 공지(수강신청 등)를 찾기 때문."""
    today = today or date.today()
    return Term(today.year, "1" if today.month <= 6 else "2")


def parse_title(text: str) -> Term | None:
    """공지 제목·강의계획서 개설 학기에서 학기를 읽는다. 연도가 없으면 None(시기 무관)."""
    if m := _FULL.search(text) or _FULL_SEASONAL.search(text):
        return Term(int(m.group(1)), _season(m.group(2)))
    if m := _YEAR_ONLY.search(text):
        return Term(int(m.group(1)))
    return None


def parse_question(text: str, today: date | None = None) -> Term | None:
    """질문이 특정 학기를 가리키면 그 학기, 아니면 None."""
    now = current_term(today)
    if m := _FULL.search(text) or _FULL_SEASONAL.search(text):
        return Term(int(m.group(1)), _season(m.group(2)))
    for offset, pattern in _RELATIVE.items():
        if pattern.search(text):
            return now.shift(offset)
    if m := _SEASON_ONLY.search(text) or _SEASONAL_ONLY.search(text):
        return Term(now.year, _season(m.group(1)))
    if m := _YEAR_ONLY.search(text):
        return Term(int(m.group(1)))
    return None
