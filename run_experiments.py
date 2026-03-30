import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import krippendorff
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import pipeline, AutoModel

from evaluate_model import evaluate_privacy_model
from utils import predict_rating, score_texts, mask_entities, MAX_LENGTH

MODEL_NAME = "gabrielloiseau/ettin-encoder-150m-privacy"
DATASET_NAME = "gabrielloiseau/privacy-200k-Mistral-Large-3"
TAB_DATASET = "ildpil/text-anonymization-benchmark"
SURVEY_DIR = Path("survey_data")
OUTPUT_FILE = Path("experiment_results.json")
SEED = 42


# -- 1. Benchmark evaluation ------------------------------------------------

def run_benchmark_evaluation(classifier):
    print("=" * 70)
    print("EXPERIMENT 1: 250-text Benchmark Evaluation")
    print("=" * 70)

    results = evaluate_privacy_model(
        lambda text: predict_rating(classifier, text),
        str(SURVEY_DIR),
        verbose=True,
    )
    preds = results["predictions"]
    return {
        "human_avg_agreement": float(results["human_avg_agreement"]),
        "human_pairwise_agreement": float(results["human_pairwise_agreement"]),
        "human_pairwise_std": float(results["human_pairwise_std"]),
        "llm_avg_agreement": float(results["llm_avg_agreement"]),
        "overall_with_llms": float(results["overall_with_llms"]),
        "overall_with_humans": float(results["overall_with_humans"]),
        "llm_agreement": {k: float(v) for k, v in results["llm_agreement"].items()},
        "dataset_breakdown": results["dataset_breakdown"],
        "predictions": preds.tolist(),
        "mean_rating": float(np.mean(preds)),
        "distribution": {str(r): float(np.mean(preds == r)) for r in range(1, 6)},
    }


# -- 2. Test-set evaluation -------------------------------------------------

def run_test_set_evaluation(classifier):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Test Set Evaluation")
    print("=" * 70)

    ds_dict = load_dataset(DATASET_NAME, "default")
    frames = []
    for split in ds_dict:
        df = ds_dict[split].to_pandas()[["text", "label"]]
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["text"].notna() & combined["label"].notna()]
    combined = combined.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    print(f"Total unique samples: {len(combined)}")

    _, test_df = train_test_split(combined, test_size=0.1, random_state=SEED, stratify=combined["label"])

    pred_arr = score_texts(classifier, test_df["text"].tolist(), batch_size=64)
    true_arr = test_df["label"].to_numpy()

    acc = accuracy_score(true_arr, pred_arr)
    macro_f1 = f1_score(true_arr, pred_arr, average="macro")
    report = classification_report(true_arr, pred_arr, output_dict=True, zero_division=0)

    print(f"  Accuracy: {acc:.4f}  |  Macro F1: {macro_f1:.4f}")

    majority = combined["label"].value_counts().idxmax()
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(f1_score(true_arr, pred_arr, average="weighted")),
        "majority_class_accuracy": float(accuracy_score(true_arr, np.full_like(true_arr, majority))),
        "total_test_samples": len(test_df),
        "total_samples": len(combined),
        "per_class": {k: v for k, v in report.items() if k not in ("accuracy", "macro avg", "weighted avg")},
    }


# -- 3. TAB pseudonymization ------------------------------------------------

def run_tab_experiment(classifier):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: TAB Pseudonymization")
    print("=" * 70)

    ds = load_dataset(TAB_DATASET, split="test")
    originals, direct_m, quasi_m, all_m = [], [], [], []
    n_direct = n_quasi = 0

    for row in ds:
        text, ents = row["text"], row["entity_mentions"]
        n_direct += sum(1 for e in ents if e.get("identifier_type") == "DIRECT")
        n_quasi += sum(1 for e in ents if e.get("identifier_type") == "QUASI")
        originals.append(text)
        direct_m.append(mask_entities(text, ents, {"DIRECT"}))
        quasi_m.append(mask_entities(text, ents, {"QUASI"}))
        all_m.append(mask_entities(text, ents, {"DIRECT", "QUASI"}))

    def _stats(scores, orig_scores=None):
        out = {"mean": float(np.mean(scores)), "std": float(np.std(scores)),
               "distribution": {str(r): float(np.mean(scores == r)) for r in range(1, 6)}}
        if orig_scores is not None:
            out["delta"] = float(np.mean(orig_scores) - np.mean(scores))
        return out

    orig_s = score_texts(classifier, originals)
    results = {
        "n_documents": len(originals), "n_direct_entities": n_direct, "n_quasi_entities": n_quasi,
        "original": _stats(orig_s),
        "mask_direct": _stats(score_texts(classifier, direct_m), orig_s),
        "mask_quasi": _stats(score_texts(classifier, quasi_m), orig_s),
        "mask_all": _stats(score_texts(classifier, all_m), orig_s),
    }
    for cond in ("original", "mask_direct", "mask_quasi", "mask_all"):
        d = results[cond]
        delta = f" (Δ={d['delta']:.2f})" if "delta" in d else ""
        print(f"  {cond:<15s}: {d['mean']:.2f}{delta}")
    return results


# -- 4. Bootstrap CIs -------------------------------------------------------

def run_bootstrap_ci(benchmark_results, n_bootstrap=1000):
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Bootstrap Confidence Intervals")
    print("=" * 70)

    predictions = np.array(benchmark_results["predictions"])
    llm_df = pd.read_csv(SURVEY_DIR / "llm_improved_combined.csv", index_col=0)
    human_df = pd.read_csv(SURVEY_DIR / "survey_results.csv")
    llm_avg = np.round(np.nanmean(llm_df.values, axis=0))
    human_avg = np.round(np.nanmean(human_df[[str(i) for i in range(250)]].values, axis=0))
    n = len(predictions)
    rng = np.random.default_rng(SEED)

    def _boot(ref):
        alphas = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n, n, replace=True)
            try:
                alphas.append(krippendorff.alpha(
                    reliability_data=np.vstack([predictions[idx], ref[idx]]),
                    level_of_measurement="ordinal",
                ))
            except Exception:
                pass
        return np.array(alphas)

    h_alphas = _boot(human_avg)
    l_alphas = _boot(llm_avg)

    results = {}
    for label, point, alphas in [("human_avg", benchmark_results["human_avg_agreement"], h_alphas),
                                  ("llm_avg", benchmark_results["llm_avg_agreement"], l_alphas)]:
        results[label] = {
            "point_estimate": float(point),
            "ci_lower": float(np.percentile(alphas, 2.5)),
            "ci_upper": float(np.percentile(alphas, 97.5)),
        }
        ci = results[label]
        print(f"  vs {label}: α = {ci['point_estimate']:.3f} [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")
    return results


# -- 5. Overlap check -------------------------------------------------------

def run_overlap_check():
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Training/Benchmark Overlap Check")
    print("=" * 70)

    with open(SURVEY_DIR / "texts.json") as f:
        bench = {" ".join(t.split()).strip().lower() for t in json.load(f).values()}

    ds_dict = load_dataset(DATASET_NAME, "default")
    n_checked = n_exact = 0
    for split in ds_dict:
        for text in ds_dict[split]["text"]:
            if text is None:
                continue
            n_checked += 1
            if " ".join(text.split()).strip().lower() in bench:
                n_exact += 1

    print(f"  Checked {n_checked} training texts, {n_exact} exact matches with benchmark")
    return {"n_benchmark": len(bench), "n_checked": n_checked, "n_exact_matches": n_exact}


# -- 6. Efficiency benchmarks -----------------------------------------------

def run_efficiency_benchmarks():
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Efficiency Benchmarks")
    print("=" * 70)

    sample = [
        "This is a short test text for benchmarking.",
        "I went to the doctor yesterday and they told me I have a condition.",
        "Just had an amazing pizza at the new restaurant downtown!",
        "My name is John Smith, I live at 123 Main Street, Springfield.",
        "The weather today is sunny with a high of 75 degrees.",
    ] * 20

    def _bench(device, bs, n_runs):
        pipe = pipeline("text-classification", model=MODEL_NAME, device=device)
        pipe(sample[:5], truncation=True, max_length=MAX_LENGTH, batch_size=bs)  # warmup
        start = time.time()
        for _ in range(n_runs):
            pipe(sample, truncation=True, max_length=MAX_LENGTH, batch_size=bs)
        return (len(sample) * n_runs) / (time.time() - start)

    eff = {}
    if torch.cuda.is_available():
        eff["gpu_throughput"] = _bench(0, 32, 3)
        print(f"  GPU: {eff['gpu_throughput']:.1f} texts/sec")
        torch.cuda.empty_cache()

    eff["cpu_throughput"] = _bench(-1, 16, 2)
    print(f"  CPU: {eff['cpu_throughput']:.1f} texts/sec")

    model = AutoModel.from_pretrained(MODEL_NAME)
    eff["n_parameters_millions"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
    print(f"  Parameters: {eff['n_parameters_millions']}M")
    return eff


# -- Main -------------------------------------------------------------------

def main():
    print(f"Loading model: {MODEL_NAME}")
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("text-classification", model=MODEL_NAME, device=device)

    results = {}
    results["benchmark"] = run_benchmark_evaluation(classifier)
    results["test_set"] = run_test_set_evaluation(classifier)
    results["tab_pseudonymization"] = run_tab_experiment(classifier)

    del classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results["bootstrap_ci"] = run_bootstrap_ci(results["benchmark"])
    results["overlap_check"] = run_overlap_check()
    results["efficiency"] = run_efficiency_benchmarks()

    def _convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, default=_convert)
    print(f"\nAll results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
