"""All settings. Values marked [paper] are stated in the paper.
Values marked [unspecified] are NOT given in the paper; they are the minimum
needed to make the system executable and can be changed from the CLI/config."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- Implementation (Section V-D) [paper] ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ollama_model: str = "llama3.2:3b"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    ollama_url: str = "http://localhost:11434"

    # --- Acceptance layer (Section IV-B) ---
    grounding_threshold: float = 0.65   # [paper] grounding(y, X) >= 0.65
    novelty_threshold: float = 0.10     # [paper] novelty(y, D_t) >= 0.10
    attribution_threshold: float = 1.0  # [unspecified] every citation must reference a retrieved doc

    # --- Evaluation (Section V) ---
    coverage_distance: float = 0.4      # [paper] relevant retrieval = distance < 0.4
    queries_per_dataset: int = 500      # [paper]
    train_size: int = 400               # [paper]
    test_size: int = 100                # [paper]
    seeds: tuple = (42, 43, 44)         # [paper]
    datasets: tuple = ("nq", "triviaqa", "hotpotqa", "stackoverflow")  # [paper]

    # --- Unspecified retrieval details ---
    top_k: int = 5                      # [unspecified] documents retrieved per query
    experience_k: int = 3               # [unspecified] past experiences injected per query
    chunk_chars: int = 500              # [unspecified] seed-corpus chunk size


DATASET_LABELS = {
    "nq": "NQ",
    "triviaqa": "TriviaQA",
    "hotpotqa": "HotpotQA",
    "stackoverflow": "StackOF",
}

SYSTEM_LABELS = {
    "standard": "Standard RAG",
    "naive": "Naive Write-back",
    "bidirectional": "Bidirectional RAG (Ours)",
}
