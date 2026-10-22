"""Small text helpers: sentence splitting and citation handling."""
import re

_CITATION = re.compile(r"\[([^\[\]\s]+)\]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def extract_citations(text: str) -> list:
    """Unique citation ids in order of appearance, e.g. '[seed-000012]'."""
    seen, out = set(), []
    for m in _CITATION.findall(text):
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def strip_citations(text: str) -> str:
    return re.sub(r"\s+", " ", _CITATION.sub("", text)).strip()


def split_sentences(text: str) -> list:
    return [s.strip() for s in _SENT_SPLIT.split(text) if re.search(r"\w", s)]
