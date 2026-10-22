"""Experience store (Section IV-C): critique logs from accepted AND rejected
responses, retrieved at query time and injected into the prompt."""
import uuid

import chromadb
import numpy as np


class ExperienceStore:
    def __init__(self):
        self._col = chromadb.EphemeralClient().create_collection(
            f"experience_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"})
        self._n = 0

    def count(self) -> int:
        return self._col.count()

    def record(self, query: str, query_embedding, accepted: bool, critique: str, stage: str = ""):
        self._n += 1
        self._col.add(ids=[f"exp-{self._n:06d}"], documents=[query],
                      embeddings=[np.asarray(query_embedding).tolist()],
                      metadatas=[{"accepted": bool(accepted), "critique": critique, "stage": stage}])

    def retrieve_notes(self, query_embedding, k: int) -> list:
        """Prompt notes from the most similar past queries (warnings and successes)."""
        n = self.count()
        if n == 0:
            return []
        res = self._col.query(query_embeddings=[np.asarray(query_embedding).tolist()],
                              n_results=min(k, n), include=["metadatas"])
        notes = []
        for m in res["metadatas"][0]:
            if m["accepted"]:
                notes.append(f"SUCCESS: a previous answer to a similar question was accepted ({m['critique']}).")
            else:
                notes.append(f"WARNING: a previous answer to a similar question was rejected. {m['critique']}")
        return notes
