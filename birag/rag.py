"""The three systems compared in the paper (Section V-B).

Forward path (Eq. 4-5):   X = R(q, D_t);  y = G(q, X)
Backward path (Eq. 6-7):  v = A(y, X, q);  D_{t+1} = W(D_t, y) if v == ACCEPT else D_t
"""
import time
from dataclasses import dataclass
from typing import Optional

from .acceptance import AcceptanceLayer, Validation
from .experience import ExperienceStore
from .text import extract_citations


@dataclass
class Result:
    query: str
    response: str
    retrieved: list
    top_distance: float
    citations: list
    latency: float
    validation: Optional[Validation] = None
    written: bool = False


def build_prompt(query: str, docs: list, notes: list) -> str:
    context = "\n".join(f"[{d.id}] {d.text}" for d in docs)
    parts = [
        "You are a question answering assistant. Answer the question using only the "
        "context documents below. After each statement, cite the supporting document "
        "by its id in square brackets, for example [seed-000001].",
        f"Context:\n{context}",
    ]
    if notes:
        parts.append("Notes from past answers:\n" + "\n".join(f"- {n}" for n in notes))
    parts.append(f"Question: {query}\nAnswer:")
    return "\n\n".join(parts)


class StandardRAG:
    """Traditional retrieve-and-generate over a static corpus."""
    name = "standard"
    performs_grounding = False

    def __init__(self, store, embedder, generator, cfg, nli=None):
        self.store, self.embedder, self.generator, self.cfg = store, embedder, generator, cfg

    def _notes(self, query_embedding) -> list:
        return []

    def _after_generation(self, query, q_emb, response, docs, learn) -> tuple:
        return None, False

    def answer(self, query: str, learn: bool) -> Result:
        t0 = time.perf_counter()
        q_emb = self.embedder.encode([query])[0]
        docs = self.store.query(q_emb, self.cfg.top_k)               # X = R(q, D_t)
        prompt = build_prompt(query, docs, self._notes(q_emb))
        response = self.generator.generate(prompt)                   # y = G(q, X)
        validation, written = self._after_generation(query, q_emb, response, docs, learn)
        return Result(query, response, docs, docs[0].distance if docs else float("inf"),
                      extract_citations(response), time.perf_counter() - t0, validation, written)


class NaiveWriteBack(StandardRAG):
    """Writes every response to the corpus without validation."""
    name = "naive"

    def _after_generation(self, query, q_emb, response, docs, learn):
        if not learn:
            return None, False
        self.store.write_back(response, self.embedder.encode([response])[0])
        return None, True


class BidirectionalRAG(StandardRAG):
    """Validated write-back (grounding -> attribution -> novelty) + experience store."""
    name = "bidirectional"
    performs_grounding = True

    def __init__(self, store, embedder, generator, cfg, nli):
        super().__init__(store, embedder, generator, cfg)
        self.acceptance = AcceptanceLayer(nli, cfg)
        self.experience = ExperienceStore()

    def _notes(self, query_embedding) -> list:
        return self.experience.retrieve_notes(query_embedding, self.cfg.experience_k)

    def _after_generation(self, query, q_emb, response, docs, learn):
        if not learn:
            return None, False
        y_emb = self.embedder.encode([response])[0]
        v = self.acceptance.validate(response, docs, self.store, y_emb)   # v = A(y, X, q)
        if v.accepted:
            self.store.write_back(response, y_emb)                        # W(D_t, y)
        # critiques of accepted and rejected responses both feed the experience store
        self.experience.record(query, q_emb, v.accepted, v.critique, v.failed_stage or "")
        return v, v.accepted


SYSTEMS = {"standard": StandardRAG, "naive": NaiveWriteBack, "bidirectional": BidirectionalRAG}
