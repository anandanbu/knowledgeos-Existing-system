"""Builds the seed corpus data/corpus/<dataset>/seed.jsonl for each dataset.

The paper does not say how its initial corpus was built, so this is a
convenience step, not part of the paper: it streams passages from the same
public HuggingFace datasets used for the queries and keeps the first N.
    python prepare_data.py                 # all four datasets, 2000 passages each
    python prepare_data.py --docs 500 --datasets nq hotpotqa
You may instead place your own .txt/.md/.jsonl files in data/corpus/<dataset>/.
"""
import argparse
import json
import re
from pathlib import Path

TAG = re.compile(r"<[^>]+>")


def _clean(t, limit=5000):
    return re.sub(r"\s+", " ", TAG.sub(" ", str(t))).strip()[:limit]


def _nq(row):
    return [row.get("answer")]


def _hotpot(row):
    ctx = row["context"]
    return [" ".join(sents) for sents in ctx["sentences"]]


def _trivia(row):
    return list(row.get("entity_pages", {}).get("wiki_context", []))


def _so(row):
    for c in ("answer_body", "answer", "body", "answers"):
        if row.get(c):
            return [row[c]]
    return []


# name -> (path, config, extractor)
SOURCES = {
    "nq": ("sentence-transformers/natural-questions", None, _nq),
    "triviaqa": ("mandarjoshi/trivia_qa", "rc", _trivia),
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor", _hotpot),
    "stackoverflow": ("koutch/stackoverflow_python", None, _so),
}


def build(name: str, n_docs: int, out_dir: Path):
    from datasets import load_dataset
    path, config, extract = SOURCES[name]
    ds = (load_dataset(path, config, split="train", streaming=True) if config
          else load_dataset(path, split="train", streaming=True))
    out_dir.mkdir(parents=True, exist_ok=True)
    seen, n = set(), 0
    with open(out_dir / "seed.jsonl", "w", encoding="utf-8") as f:
        for row in ds:
            for t in extract(row):
                t = _clean(t) if t else ""
                if len(t) > 40 and t not in seen:
                    seen.add(t)
                    f.write(json.dumps({"text": t}) + "\n")
                    n += 1
                    if n >= n_docs:
                        return n
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument("--docs", type=int, default=2000, help="passages per dataset")
    ap.add_argument("--out", default="data/corpus")
    a = ap.parse_args()
    for name in a.datasets:
        n = build(name, a.docs, Path(a.out) / name)
        print(f"{name}: wrote {n} passages to {a.out}/{name}/seed.jsonl")
        if n == 0:
            print(f"  WARNING: no passages extracted for {name}; add your own files to {a.out}/{name}/")


if __name__ == "__main__":
    main()
