"""Model backends exactly as listed in Section V-D:
  - Embeddings : all-MiniLM-L6-v2 (sentence-transformers)
  - Generator  : Ollama llama3.2:3b (local inference)
  - Grounding  : cross-encoder/nli-deberta-v3-base
Each exposes a tiny interface so tests can substitute fakes:
  embedder.encode(list[str]) -> ndarray (L2-normalised)
  generator.generate(prompt: str) -> str
  nli.entail_probs(list[(premise, hypothesis)]) -> list[float]
"""
import numpy as np
import requests


class SentenceTransformerEmbedder:
    def __init__(self, name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name)

    def encode(self, texts):
        return self.model.encode(list(texts), normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True)


class OllamaGenerator:
    def __init__(self, model: str, url: str = "http://localhost:11434", timeout: int = 600):
        self.model, self.url, self.timeout = model, url.rstrip("/"), timeout
        self._check()

    def _check(self):
        try:
            tags = requests.get(f"{self.url}/api/tags", timeout=10).json()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.url}. Start it with `ollama serve`.") from e
        names = [m.get("name", "") for m in tags.get("models", [])]
        if not any(n == self.model or n.startswith(self.model + ":") or
                   n.split(":")[0] == self.model for n in names):
            raise RuntimeError(f"Model '{self.model}' not found in Ollama. "
                               f"Run `ollama pull {self.model}`. Installed: {names}")

    def generate(self, prompt: str) -> str:
        r = requests.post(f"{self.url}/api/generate",
                          json={"model": self.model, "prompt": prompt, "stream": False},
                          timeout=self.timeout)
        r.raise_for_status()
        return r.json()["response"].strip()


class CrossEncoderNLI:
    """P_NLI(entail | premise, hypothesis) from DeBERTa-v3-base NLI cross-encoder."""

    def __init__(self, name: str, batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(self.device).eval()
        labels = {int(i): l.lower() for i, l in self.model.config.id2label.items()}
        self.entail_idx = next(i for i, l in labels.items() if l == "entailment")
        self.batch_size = batch_size

    def entail_probs(self, pairs):
        out = []
        torch = self.torch
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            enc = self.tok([p for p, _ in batch], [h for _, h in batch], padding=True,
                           truncation="only_first", max_length=512,
                           return_tensors="pt").to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits, dim=-1)
            out.extend(probs[:, self.entail_idx].cpu().numpy().tolist())
        return out
