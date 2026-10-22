"""Retrieval corpus D_t backed by ChromaDB (cosine distance), seed ingestion,
and the write-back operator W."""
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np

from .text import split_sentences

_BATCH = 1000


@dataclass(frozen=True)
class Hit:
    id: str
    text: str
    distance: float
    source: str  # "seed" or "generated"


@dataclass
class SeedCorpus:
    ids: list
    texts: list
    embeddings: np.ndarray


def chunk_text(text: str, max_chars: int) -> list:
    """One chunk per paragraph; paragraphs longer than max_chars are split on
    sentence boundaries into chunks of at most max_chars."""
    chunks = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        cur = ""
        for sent in split_sentences(para):
            if cur and len(cur) + 1 + len(sent) > max_chars:
                chunks.append(cur)
                cur = sent
            else:
                cur = f"{cur} {sent}".strip()
        if cur:
            chunks.append(cur)
    return chunks


def load_seed_corpus(directory, embedder, max_chars: int = 500) -> SeedCorpus:
    """One-time document ingestion (.txt / .md / .jsonl with a 'text' field)."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Seed corpus directory not found: {directory}")
    texts = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() in (".txt", ".md"):
            texts.extend(chunk_text(path.read_text(encoding="utf-8", errors="ignore"), max_chars))
        elif path.suffix.lower() == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    texts.extend(chunk_text(json.loads(line)["text"], max_chars))
    if not texts:
        raise ValueError(f"No ingestible documents (.txt/.md/.jsonl) in {directory}")
    ids = [f"seed-{i:06d}" for i in range(len(texts))]
    return SeedCorpus(ids, texts, np.asarray(embedder.encode(texts), dtype=np.float32))


class VectorStore:
    """D_t. A fresh instance is built per experiment from the shared seed corpus."""

    def __init__(self, embedder, seed: SeedCorpus = None):
        self.embedder = embedder
        self._col = chromadb.EphemeralClient().create_collection(
            f"corpus_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"})
        self._generated = 0
        if seed is not None:
            for i in range(0, len(seed.ids), _BATCH):
                s = slice(i, i + _BATCH)
                self._col.add(ids=seed.ids[s], documents=seed.texts[s],
                              embeddings=seed.embeddings[s].tolist(),
                              metadatas=[{"source": "seed"}] * len(seed.ids[s]))

    def count(self) -> int:
        return self._col.count()

    def generated_count(self) -> int:
        return self._generated

    def composition_ratio(self) -> float:
        """alpha(D_t): fraction of model-generated content."""
        n = self.count()
        return self._generated / n if n else 0.0

    def query(self, embedding, k: int) -> list:
        n = self.count()
        if n == 0:
            return []
        res = self._col.query(query_embeddings=[np.asarray(embedding).tolist()],
                              n_results=min(k, n),
                              include=["documents", "distances", "metadatas"])
        return [Hit(i, d, float(dist), m["source"]) for i, d, dist, m in
                zip(res["ids"][0], res["documents"][0], res["distances"][0], res["metadatas"][0])]

    def write_back(self, text: str, embedding) -> str:
        """Write-back operator W(D_t, y)."""
        self._generated += 1
        doc_id = f"gen-{self._generated:06d}"
        self._col.add(ids=[doc_id], documents=[text],
                      embeddings=[np.asarray(embedding).tolist()],
                      metadatas=[{"source": "generated"}])
        return doc_id
