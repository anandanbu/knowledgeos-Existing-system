"""Query streams for the four datasets (Section V-A): 500 queries per dataset
per seed, split 400 train / 100 test.

Source order for each dataset:
  1. <datasets_dir>/<name>.jsonl  (one JSON object per line with a "question" field)
  2. HuggingFace `datasets` (column picked from the candidates below)
"""
import json
import random
from pathlib import Path

# name -> (hf path, hf config, split, candidate question columns)
HF_SOURCES = {
    "nq": ("sentence-transformers/natural-questions", None, "train", ["query", "question"]),
    "triviaqa": ("mandarjoshi/trivia_qa", "rc.nocontext", "train", ["question"]),
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor", "train", ["question"]),
    "stackoverflow": ("koutch/stackoverflow_python", None, "train",
                      ["question_title", "title", "question", "query"]),
}


def _load_local(path: Path) -> list:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line).get("question")
            if q:
                out.append(q.strip())
    return out


def _load_hf(name: str, seed: int, n: int) -> list:
    from datasets import load_dataset
    path, config, split, columns = HF_SOURCES[name]
    ds = load_dataset(path, config, split=split) if config else load_dataset(path, split=split)
    col = next((c for c in columns if c in ds.column_names), None)
    if col is None:
        raise KeyError(f"None of {columns} found in {path}; columns: {ds.column_names}")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [str(q).strip() for q in ds[col] if str(q).strip()]


def load_queries(name: str, seed: int, n: int, datasets_dir=None) -> list:
    """Deterministic sample of n queries for (dataset, seed)."""
    local = Path(datasets_dir) / f"{name}.jsonl" if datasets_dir else None
    if local is not None and local.exists():
        pool = _load_local(local)
        random.Random(seed).shuffle(pool)
        queries = pool[:n]
    else:
        queries = _load_hf(name, seed, n)
    if len(queries) < n:
        raise ValueError(f"{name}: need {n} queries, found {len(queries)}")
    return queries


def split_queries(queries: list, train_size: int, test_size: int):
    return queries[:train_size], queries[train_size:train_size + test_size]
