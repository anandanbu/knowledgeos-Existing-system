"""Produces Table I, Table II, Fig. 1 and Fig. 2 from the paper, plus raw runs."""
import csv
import json
from pathlib import Path

import numpy as np

from .config import DATASET_LABELS, SYSTEM_LABELS

SYSTEM_ORDER = ["standard", "naive", "bidirectional"]
COLORS = {"standard": "#e8685f", "naive": "#5bd58a", "bidirectional": "#5aa9e6"}


def _of(runs, system, key, dataset=None):
    return [r[key] for r in runs if r["system"] == system and (dataset is None or r["dataset"] == dataset)]


def table1(runs) -> list:
    rows = []
    for s in SYSTEM_ORDER:
        if not _of(runs, s, "coverage"):
            continue
        cov, cit = _of(runs, s, "coverage"), _of(runs, s, "citation_f1")
        rows.append({
            "System": SYSTEM_LABELS[s],
            "Coverage (%)": f"{np.mean(cov):.2f} ± {np.std(cov):.2f}",
            "Growth (docs)": f"{np.mean(_of(runs, s, 'growth')):.0f}",
            "Grounding": "Yes" if any(_of(runs, s, "grounding")) else "No",
            "Citation F1 (%)": f"{np.mean(cit):.2f} ± {np.std(cit):.2f}",
            "Latency (s)": f"{np.mean(_of(runs, s, 'latency')):.1f}",
            "Experiments": len(cov),
        })
    return rows


def table2(runs, datasets) -> list:
    rows = []
    for s in SYSTEM_ORDER:
        if not _of(runs, s, "coverage"):
            continue
        row = {"System": SYSTEM_LABELS[s].replace(" (Ours)", "")}
        for d in datasets:
            row[DATASET_LABELS[d]] = f"{np.mean(_of(runs, s, 'coverage', d)):.1f}"
        rows.append(row)
    return rows


def _markdown(title, rows) -> str:
    cols = list(rows[0])
    lines = [f"### {title}", "", "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _figures(runs, datasets, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    systems = [s for s in SYSTEM_ORDER if _of(runs, s, "coverage")]
    labels = [SYSTEM_LABELS[s].replace(" (Ours)", "") for s in systems]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, key, title, ylabel in [(ax[0], "coverage", "Average Coverage Across Datasets", "Coverage (%)"),
                                  (ax[1], "growth", "Average Corpus Growth Across Datasets",
                                   "Corpus Growth (Documents)")]:
        vals = [np.mean(_of(runs, s, key)) for s in systems]
        bars = a.bar(labels, vals, color=[COLORS[s] for s in systems])
        a.bar_label(bars, labels=[f"{v:.1f}" for v in vals])
        a.set_title(title)
        a.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out / "fig1_coverage_and_growth.png", dpi=150)
    plt.close(fig)

    fig, a = plt.subplots(figsize=(9, 4.5))
    w = 0.8 / len(systems)
    x = np.arange(len(datasets))
    for i, s in enumerate(systems):
        vals = [np.mean(_of(runs, s, "coverage", d)) for d in datasets]
        bars = a.bar(x + i * w, vals, w, label=SYSTEM_LABELS[s].replace(" (Ours)", ""), color=COLORS[s])
        a.bar_label(bars, labels=[f"{v:.1f}%" for v in vals], fontsize=7)
    a.set_xticks(x + w * (len(systems) - 1) / 2)
    a.set_xticklabels([DATASET_LABELS[d] for d in datasets])
    a.set_ylabel("Coverage (%)")
    a.set_ylim(0, 105)
    a.set_title("Coverage by Dataset and System")
    a.legend()
    fig.tight_layout()
    fig.savefig(out / "fig2_coverage_by_dataset.png", dpi=150)
    plt.close(fig)


def write_report(runs, datasets, out_dir) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw_runs.json").write_text(json.dumps(runs, indent=2))
    t1, t2 = table1(runs), table2(runs, datasets)
    for name, rows in (("table1_main_results.csv", t1), ("table2_coverage_by_dataset.csv", t2)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    md = (_markdown("TABLE I: Main results (mean ± std across all experiments)", t1) + "\n" +
          _markdown("TABLE II: Coverage (%) by dataset and system", t2))
    bi = [r for r in runs if r["system"] == "bidirectional"]
    if bi:
        n_train_acc = sum(r["accepted"] for r in bi)
        md += (f"\nBidirectional RAG accepted {n_train_acc} candidate responses across {len(bi)} runs "
               f"(mean growth {np.mean([r['growth'] for r in bi]):.0f} docs/run; mean generated-content "
               f"ratio α = {np.mean([r['composition_ratio'] for r in bi]):.3f}).\n")
    (out / "results.md").write_text(md)
    _figures(runs, datasets, out)
    return md
