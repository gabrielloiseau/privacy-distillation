import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import krippendorff


def load_evaluation_data(data_dir: str = "survey_data") -> dict[str, Any]:
    """Load benchmark texts, LLM ratings, and human survey ratings."""
    p = Path(data_dir)
    with open(p / "texts.json") as f:
        texts = json.load(f)
    return {
        "texts": texts,
        "llm_ratings": pd.read_csv(p / "llm_improved_combined.csv", index_col=0),
        "human_ratings": pd.read_csv(p / "survey_results.csv"),
        "text_idx": json.load(open(p / "text_idx.json")),
    }


def _alpha(a: np.ndarray, b: np.ndarray) -> float:
    """Ordinal Krippendorff's alpha between two rating vectors."""
    combined = np.vstack([a, b])
    valid = ~np.isnan(combined[0]) & ~np.isnan(combined[1])
    if valid.sum() < 2:
        return np.nan
    try:
        return krippendorff.alpha(reliability_data=combined[:, valid], level_of_measurement="ordinal")
    except Exception:
        return np.nan


def _alpha_group(predictions: np.ndarray, group: np.ndarray) -> float:
    """Alpha when adding predictions as a new rater to an existing group."""
    try:
        return krippendorff.alpha(
            reliability_data=np.vstack([predictions.reshape(1, -1), group]),
            level_of_measurement="ordinal",
        )
    except Exception:
        return np.nan


def evaluate_privacy_model(
    predict_fn: Callable[[str], int],
    data_dir: str = "survey_data",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Evaluate a privacy prediction model against LLM and human baselines.

    Args:
        predict_fn: Takes a text string, returns a privacy rating (1-5).
        data_dir:   Path to directory containing evaluation data.
        verbose:    Print progress and results.

    Returns dict with predictions, agreement scores, and per-dataset breakdown.
    """
    data = load_evaluation_data(data_dir)
    texts, llm_df, human_df, text_idx = data["texts"], data["llm_ratings"], data["human_ratings"], data["text_idx"]

    llm_matrix = llm_df.values
    human_matrix = human_df[[str(i) for i in range(250)]].values

    # Predictions
    n = len(texts)
    predictions = np.zeros(n)
    for i in range(n):
        try:
            predictions[i] = max(1, min(5, int(round(predict_fn(texts[str(i)])))))
        except Exception as e:
            if verbose:
                print(f"  Warning: text {i}: {e}")
            predictions[i] = np.nan
        if verbose and (i + 1) % 50 == 0:
            print(f"  Predicted {i + 1}/{n}")

    llm_avg = np.round(np.nanmean(llm_matrix, axis=0))
    human_avg = np.round(np.nanmean(human_matrix, axis=0))

    # Per-LLM agreement
    llm_agreement = {name: _alpha(predictions, np.round(llm_matrix[i])) for i, name in enumerate(llm_df.index)}

    # Pairwise with humans (sample up to 500)
    np.random.seed(42)
    sample = np.random.choice(human_matrix.shape[0], min(500, human_matrix.shape[0]), replace=False)
    pw = [a for j in sample if not np.isnan(a := _alpha(predictions, human_matrix[j]))]

    results = {
        "predictions": predictions,
        "llm_agreement": llm_agreement,
        "llm_avg_agreement": _alpha(predictions, llm_avg),
        "human_avg_agreement": _alpha(predictions, human_avg),
        "human_pairwise_agreement": np.mean(pw) if pw else np.nan,
        "human_pairwise_std": np.std(pw) if pw else np.nan,
        "overall_with_llms": _alpha_group(predictions, np.round(llm_matrix)),
        "overall_with_humans": _alpha_group(predictions, human_matrix),
        "dataset_breakdown": {
            ds: {
                "model_mean": float(np.nanmean(predictions[idx])),
                "human_mean": float(np.nanmean(human_avg[idx])),
                "llm_mean": float(np.nanmean(llm_avg[idx])),
            }
            for ds, idx in text_idx.items()
        },
    }

    if verbose:
        _print_results(results, list(llm_df.index))
    return results


def _print_results(results: dict, llm_names: list[str]):
    preds = results["predictions"][~np.isnan(results["predictions"])]
    print(f"\nPredictions: mean={np.mean(preds):.2f}, std={np.std(preds):.2f}")
    print(f"  Distribution: " + " ".join(f"{r}:{100*np.mean(preds==r):.0f}%" for r in range(1, 6)))

    print(f"\nAgreement with LLM avg:    α = {results['llm_avg_agreement']:.3f}")
    print(f"Agreement with Human avg:  α = {results['human_avg_agreement']:.3f}")
    print(f"Pairwise with humans:      α = {results['human_pairwise_agreement']:.3f} (±{results['human_pairwise_std']:.3f})")

    print(f"\n{'Dataset':<25s} | {'Model':>6s} | {'Human':>6s} | {'LLM':>6s}")
    print("-" * 55)
    for ds, s in results["dataset_breakdown"].items():
        print(f"{ds:<25s} | {s['model_mean']:6.2f} | {s['human_mean']:6.2f} | {s['llm_mean']:6.2f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a privacy model on the 250-text benchmark")
    parser.add_argument("--hf-model", required=True, help="HuggingFace model name for text classification")
    parser.add_argument("--data-dir", default="survey_data", help="Path to survey_data directory")
    args = parser.parse_args()

    from transformers import pipeline
    from utils import predict_rating

    classifier = pipeline("text-classification", model=args.hf_model)
    evaluate_privacy_model(lambda text: predict_rating(classifier, text), args.data_dir)


if __name__ == "__main__":
    main()
