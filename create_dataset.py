import argparse
import json
import os
from pathlib import Path

import dspy
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

from annotator import annotator_predict

TEACHER_MODEL = "mistral/mistral-large-latest"
SAMPLE_SIZE = 20_000
SEED = 42
DATA_DIR = Path("data")

# --------------------------------------------------------------------------
# Source dataset definitions
# --------------------------------------------------------------------------
# Each entry: (short_name, hf_id, hf_config, split, text_column, notes)
# MHB is not on HuggingFace; must provide it manually (see README).

SOURCES = {
    "BAC": {
        "hf_id": "tasksource/blog_authorship_corpus",
        "text_col": "text",
    },
    "EE": {
        "hf_id": "sujan-maharjan/enron_email_dataset",
        "text_col": "text",
    },
    "MQ": {
        "hf_id": "Malikeh1375/medical-question-answering-datasets",
        "text_col": "question",
    },
    "RC": {
        "hf_id": "SocialGrep/one-million-reddit-confessions",
        "text_col": "body",
    },
    "RLA": {
        "hf_id": "jonathanli/legal-advice-reddit",
        "text_col": "text",
    },
    "RMHP": {
        "hf_id": "solomonk/reddit_mental_health_posts",
        "text_col": "text",
    },
    "TR": {
        "hf_id": "Kerassy/trustpilot-reviews-123k",
        "text_col": "review",
    },
    "TW": {
        "hf_id": "enryu43/twitter100m_tweets",
        "text_col": "text",
    },
    "YR": {
        "hf_id": "yashraizad/yelp-open-dataset-reviews",
        "text_col": "text",
    },
    # MHB is not hosted on HuggingFace.
    # Place a CSV/parquet with a "text" column at data/MHB/raw_texts.parquet
    # and the script will pick it up automatically.
    "MHB": {
        "hf_id": None,
        "text_col": "text",
    },
}


# --------------------------------------------------------------------------
# Dataset loading & sampling
# --------------------------------------------------------------------------

def load_and_sample(domain: str, n: int = SAMPLE_SIZE, benchmark_texts: set[str] | None = None) -> list[str]:
    """Load a source dataset from HuggingFace and return *n* sampled texts.

    Texts that appear in the 250-text human benchmark are excluded before sampling.
    """
    cfg = SOURCES[domain]
    hf_id = cfg["hf_id"]
    text_col = cfg["text_col"]

    if hf_id is None:
        raw_path = DATA_DIR / domain / "raw_texts.parquet"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"{domain} is not on HuggingFace. Place a parquet file with a "
                f"'{text_col}' column at {raw_path}"
            )
        df = pd.read_parquet(raw_path)
        texts = df[text_col].dropna().tolist()
    else:
        print(f"  Loading {hf_id} from HuggingFace...")
        ds = load_dataset(hf_id, trust_remote_code=True)
        if isinstance(ds, dict):
            all_texts = []
            for split in ds.values():
                all_texts.extend(split[text_col])
            texts = all_texts
        else:
            texts = ds[text_col]

    texts = [t for t in texts if t and isinstance(t, str) and len(t.strip()) > 10]

    if benchmark_texts:
        norm = lambda s: " ".join(s.split()).strip().lower()
        bench_norm = {norm(t) for t in benchmark_texts}
        texts = [t for t in texts if norm(t) not in bench_norm]

    rng = np.random.default_rng(SEED)
    if len(texts) > n:
        indices = rng.choice(len(texts), size=n, replace=False)
        texts = [texts[i] for i in sorted(indices)]
    print(f"  {domain}: sampled {len(texts)} texts")
    return texts


# --------------------------------------------------------------------------
# LLM annotation
# --------------------------------------------------------------------------

def annotate_texts(texts: list[str], lm) -> pd.DataFrame:
    """Annotate a list of texts with privacy ratings using the teacher LLM.

    Returns a DataFrame with columns [text, label].
    """
    annotator_predict.set_lm(lm)
    records = []
    for text in tqdm(texts, desc="Annotating"):
        try:
            result = annotator_predict(user_text=text)
            rating_str = result.privacy_rating
            rating = int(rating_str) if rating_str.isdigit() else None
            if rating and 1 <= rating <= 5:
                records.append({"text": text, "label": rating})
            else:
                records.append({"text": text, "label": pd.NA})
        except Exception as e:
            print(f"  Error: {e}")
            records.append({"text": text, "label": pd.NA})
    df = pd.DataFrame(records)
    df["label"] = df["label"].astype("Int64")
    return df


# --------------------------------------------------------------------------
# Push to HuggingFace Hub
# --------------------------------------------------------------------------

def push_to_hub(data_dir: Path, repo_id: str):
    """Push per-domain parquet files to HuggingFace as a multi-config dataset."""
    configs: dict[str, Dataset] = {}
    for domain_dir in sorted(data_dir.iterdir()):
        parquet = domain_dir / "annotations.parquet"
        if not parquet.exists():
            continue
        df = pd.read_parquet(parquet)[["text", "label"]].dropna()
        df = df.reset_index(drop=True)
        configs[domain_dir.name] = Dataset.from_pandas(df, preserve_index=False)
        print(f"  {domain_dir.name}: {len(df)} annotated texts")

    if not configs:
        raise ValueError(f"No annotation parquet files found under {data_dir}")

    DatasetDict(configs).push_to_hub(repo_id)
    print(f"\nPushed {len(configs)} configs to https://huggingface.co/datasets/{repo_id}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Create the privacy-annotated dataset")
    parser.add_argument("--domains", nargs="+", default=list(SOURCES.keys()),
                        help="Domains to process (default: all)")
    parser.add_argument("--teacher-model", default=TEACHER_MODEL,
                        help="DSPy LM identifier for the teacher model")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--push", action="store_true", help="Push dataset to HuggingFace Hub")
    parser.add_argument("--repo-id", type=str, help="HuggingFace repo id (required with --push)")
    args = parser.parse_args()

    if args.push:
        if not args.repo_id:
            parser.error("--repo-id is required when using --push")
        push_to_hub(args.data_dir, args.repo_id)
        return

    # Load benchmark texts to exclude from sampling
    bench_path = Path("survey_data/texts.json")
    benchmark_texts = set()
    if bench_path.exists():
        with open(bench_path) as f:
            benchmark_texts = set(json.load(f).values())
        print(f"Loaded {len(benchmark_texts)} benchmark texts to exclude")

    lm = dspy.LM(args.teacher_model)

    for domain in args.domains:
        if domain not in SOURCES:
            print(f"Unknown domain '{domain}', skipping")
            continue

        out_dir = args.data_dir / domain
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "annotations.parquet"

        # Resume: load existing annotations to skip already-annotated texts
        existing = pd.DataFrame()
        if out_path.exists():
            existing = pd.read_parquet(out_path)
            done = set(existing["text"].tolist())
            print(f"  {domain}: {len(existing)} existing annotations found, resuming...")
        else:
            done = set()

        print(f"\nProcessing {domain}...")
        texts = load_and_sample(domain, n=args.sample_size, benchmark_texts=benchmark_texts)

        remaining = [t for t in texts if t not in done]
        if not remaining:
            print(f"  {domain}: all {len(texts)} texts already annotated")
            continue

        print(f"  {domain}: {len(remaining)} texts to annotate ({len(done)} already done)")
        new_df = annotate_texts(remaining, lm)

        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["text"], keep="last").reset_index(drop=True)
        combined.to_parquet(out_path, index=False)

        valid = combined["label"].notna().sum()
        print(f"  {domain}: saved {len(combined)} texts ({valid} annotated) to {out_path}")

    print("\nDone. To push to HuggingFace Hub:")
    print(f"  python create_dataset.py --push --repo-id your-username/your-dataset")


if __name__ == "__main__":
    main()
