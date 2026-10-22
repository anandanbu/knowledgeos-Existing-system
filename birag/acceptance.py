"""Multi-stage acceptance layer A(y, X, q) from Section IV-B.
Three sequential checks: grounding (NLI) -> attribution -> novelty."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .text import extract_citations, split_sentences, strip_citations


@dataclass
class Validation:
    accepted: bool
    grounding: Optional[float] = None
    attribution: Optional[float] = None
    novelty: Optional[float] = None
    failed_stage: Optional[str] = None
    critique: str = ""


class AcceptanceLayer:
    def __init__(self, nli, cfg):
        self.nli, self.cfg = nli, cfg

    # Eq. (8): mean over sentences of max entailment probability over retrieved docs
    def grounding(self, response: str, docs: list) -> float:
        sentences = split_sentences(strip_citations(response))
        if not sentences or not docs:
            return 0.0
        pairs = [(d.text, s) for s in sentences for d in docs]
        probs = np.asarray(self.nli.entail_probs(pairs), dtype=float).reshape(len(sentences), len(docs))
        return float(probs.max(axis=1).mean())

    # Eq. (9): |citations(y) ∩ IDs(X)| / |citations(y)|
    def attribution(self, response: str, docs: list) -> float:
        cites = extract_citations(response)
        if not cites:
            return 0.0
        ids = {d.id for d in docs}
        return len([c for c in cites if c in ids]) / len(cites)

    # Eq. (10): 1 - max cosine similarity to any corpus document.
    # The store uses cosine distance (= 1 - similarity), so the nearest
    # document's distance is exactly 1 - max sim.
    def novelty(self, response_embedding, store) -> float:
        hits = store.query(response_embedding, 1)
        return 1.0 if not hits else float(hits[0].distance)

    def validate(self, response: str, docs: list, store, response_embedding) -> Validation:
        cfg = self.cfg
        g = self.grounding(response, docs)
        if g < cfg.grounding_threshold:
            return Validation(False, g, failed_stage="grounding", critique=(
                f"Grounding {g:.2f} is below {cfg.grounding_threshold:.2f}: statements "
                f"were not entailed by the retrieved documents."))
        a = self.attribution(response, docs)
        if a < cfg.attribution_threshold:
            return Validation(False, g, a, failed_stage="attribution", critique=(
                f"Attribution {a:.2f} is below {cfg.attribution_threshold:.2f}: citations "
                f"must reference retrieved document ids."))
        n = self.novelty(response_embedding, store)
        if n < cfg.novelty_threshold:
            return Validation(False, g, a, n, failed_stage="novelty", critique=(
                f"Novelty {n:.2f} is below {cfg.novelty_threshold:.2f}: answer duplicates "
                f"existing corpus content."))
        return Validation(True, g, a, n, critique=(
            f"grounding={g:.2f}, attribution={a:.2f}, novelty={n:.2f}"))
