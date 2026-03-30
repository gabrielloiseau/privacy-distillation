import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import pipeline

from utils import score_texts, mask_entities, randomly_mask_text

MODEL_NAME = "gabrielloiseau/ettin-encoder-150m-privacy"
TAB_DATASET = "ildpil/text-anonymization-benchmark"
SEED = 42
OUTPUT_FILE = Path("tab_experiment_results.json")


def run_tab_experiment(classifier):
    ds = load_dataset(TAB_DATASET, split="test")
    print(f"Loaded {len(ds)} TAB documents")

    originals, direct_m, quasi_m, all_m, random_m = [], [], [], [], []
    n_direct = n_quasi = 0

    for i, row in enumerate(ds):
        text, ents = row["text"], row["entity_mentions"]
        n_direct += sum(1 for e in ents if e.get("identifier_type") == "DIRECT")
        n_quasi += sum(1 for e in ents if e.get("identifier_type") == "QUASI")
        originals.append(text)
        direct_m.append(mask_entities(text, ents, {"DIRECT"}))
        quasi_m.append(mask_entities(text, ents, {"QUASI"}))
        all_m.append(mask_entities(text, ents, {"DIRECT", "QUASI"}))
        random_m.append(randomly_mask_text(text, fraction=0.3, seed=SEED + i))

    print(f"DIRECT entities: {n_direct}  |  QUASI entities: {n_quasi}")

    def _stats(scores, orig_scores=None):
        out = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "pct_harmless": float(np.mean(scores == 1)) * 100,
            "distribution": {str(r): float(np.mean(scores == r)) for r in range(1, 6)},
        }
        if orig_scores is not None:
            out["delta"] = float(np.mean(orig_scores) - np.mean(scores))
        return out

    orig_s = score_texts(classifier, originals)
    results = {
        "n_documents": len(originals),
        "n_direct_entities": n_direct,
        "n_quasi_entities": n_quasi,
        "original": _stats(orig_s),
        "mask_direct": _stats(score_texts(classifier, direct_m), orig_s),
        "mask_quasi": _stats(score_texts(classifier, quasi_m), orig_s),
        "mask_all": _stats(score_texts(classifier, all_m), orig_s),
        "mask_random_30pct": _stats(score_texts(classifier, random_m), orig_s),
    }

    for cond in ("original", "mask_direct", "mask_quasi", "mask_all", "mask_random_30pct"):
        d = results[cond]
        delta = f" (Δ={d['delta']:.2f})" if "delta" in d else ""
        tag = " [sanity]" if cond == "mask_random_30pct" else ""
        print(f"  {cond:<20s}: {d['mean']:.2f}{delta}  | harmless: {d['pct_harmless']:.1f}%{tag}")

    return results


def main():
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("text-classification", model=MODEL_NAME, device=device)
    results = run_tab_experiment(classifier)

    with open(OUTPUT_FILE, "w") as f:
        json.dump({"tab_pseudonymization": results}, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
