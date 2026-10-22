"""Experiment protocol: 4 datasets x 3 seeds = 12 experiments per system.

Per experiment (fresh corpus copy per system):
  1. TRAIN: stream train queries through the system with write-back enabled
            -> corpus growth, latency, acceptance statistics
  2. TEST : stream test queries with write-back disabled
            -> coverage (top-1 distance < 0.4), citation F1
"""
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .config import SYSTEM_LABELS
from .corpus import VectorStore, load_seed_corpus
from .data import load_queries, split_queries
from .rag import SYSTEMS
from .text import extract_citations


@dataclass
class Components:
    embedder: object
    generator: object
    nli: object


def citation_f1(response: str, docs: list) -> float:
    """Per-response harmonic mean of citation precision and recall.
    precision = |cited ∩ retrieved| / |cited|;  recall = |cited ∩ retrieved| / |retrieved|."""
    cited = set(extract_citations(response))
    retrieved = {d.id for d in docs}
    if not cited or not retrieved:
        return 0.0
    hit = len(cited & retrieved)
    p, r = hit / len(cited), hit / len(retrieved)
    return 0.0 if hit == 0 else 2 * p * r / (p + r)


def run_single(system_key, dataset, seed, train_q, test_q, seed_corpus, comps, cfg) -> dict:
    store = VectorStore(comps.embedder, seed_corpus)
    system = SYSTEMS[system_key](store, comps.embedder, comps.generator, cfg, comps.nli)
    initial_docs = store.count()

    train = [system.answer(q, learn=True) for q in train_q]
    test = [system.answer(q, learn=False) for q in test_q]

    stages = Counter(r.validation.failed_stage for r in train
                     if r.validation is not None and not r.validation.accepted)
    return {
        "system": system_key, "dataset": dataset, "seed": seed,
        "coverage": 100.0 * float(np.mean([r.top_distance < cfg.coverage_distance for r in test])),
        "growth": store.count() - initial_docs,
        "grounding": system.performs_grounding,
        "citation_f1": 100.0 * float(np.mean([citation_f1(r.response, r.retrieved) for r in test])),
        "latency": float(np.mean([r.latency for r in train])),
        "composition_ratio": store.composition_ratio(),
        "accepted": sum(1 for r in train if r.written) if system_key == "bidirectional" else None,
        "rejected_by_stage": dict(stages) if system_key == "bidirectional" else None,
    }


def run_all(cfg, comps, corpus_dir, datasets_dir=None, systems=None, log=print) -> list:
    systems = systems or list(SYSTEMS)
    runs = []
    for dataset in cfg.datasets:
        seed_corpus = load_seed_corpus(f"{corpus_dir}/{dataset}", comps.embedder, cfg.chunk_chars)
        log(f"[{dataset}] seed corpus: {len(seed_corpus.ids)} chunks")
        for seed in cfg.seeds:
            queries = load_queries(dataset, seed, cfg.queries_per_dataset, datasets_dir)
            train_q, test_q = split_queries(queries, cfg.train_size, cfg.test_size)
            for key in systems:
                t0 = time.perf_counter()
                run = run_single(key, dataset, seed, train_q, test_q, seed_corpus, comps, cfg)
                runs.append(run)
                log(f"[{dataset} seed={seed}] {SYSTEM_LABELS[key]:<26} coverage={run['coverage']:.1f}% "
                    f"growth={run['growth']} citF1={run['citation_f1']:.1f}% "
                    f"({time.perf_counter() - t0:.0f}s)")
    return runs
