import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

DATASET_NAME = "gabrielloiseau/privacy-200k-Mistral-Large-3"


def load_data(dataset_name: str = DATASET_NAME) -> pd.DataFrame:
    """Load and combine all splits from the HuggingFace dataset."""
    ds_dict = load_dataset(dataset_name, "default")
    frames = []
    for split in ds_dict:
        df = ds_dict[split].to_pandas()[["text", "label"]]
        df = df[df["text"].notna() & df["label"].notna()]
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    print(f"Loaded {len(combined)} unique samples from {dataset_name}")
    print(f"Label distribution:\n{combined['label'].value_counts().sort_index()}")
    return combined


def compute_class_weights(labels) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced data."""
    counts = np.bincount([int(l) for l in labels], minlength=5)
    weights = len(labels) / (5 * counts + 1e-6)
    weights = weights / weights.sum() * 5
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(eval_pred) -> dict:
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=list(range(5)), zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }
    for i in range(5):
        metrics[f"f1_class_{i + 1}"] = f1[i] if i < len(f1) else 0.0
    return metrics


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        if self.class_weights is not None:
            loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=5, problem_type="single_label_classification",
    )

    df = load_data(args.dataset)

    # Convert labels to 0-indexed
    df = df.copy()
    df["label"] = df["label"] - 1

    # 90 / 5 / 5 split
    train_val_df, test_df = train_test_split(df, test_size=0.05, random_state=args.seed, stratify=df["label"])
    adj_val = 0.05 / 0.95
    train_df, val_df = train_test_split(train_val_df, test_size=adj_val, random_state=args.seed, stratify=train_val_df["label"])

    def tokenize(ds):
        return ds.map(
            lambda ex: tokenizer(ex["text"], truncation=True, max_length=args.max_length, padding="max_length"),
            batched=True, remove_columns=["text"],
        )

    train_ds = tokenize(Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False))
    val_ds = tokenize(Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False))
    test_ds = tokenize(Dataset.from_pandas(test_df[["text", "label"]], preserve_index=False))
    print(f"Splits — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    class_weights = compute_class_weights(train_ds["label"]) if args.class_weights else None

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=3,
        seed=args.seed,
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    model_path = output_dir / "model"
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))

    test_results = trainer.evaluate(test_ds)
    predictions = trainer.predict(test_ds)
    test_preds = np.argmax(predictions.predictions, axis=1) + 1
    test_labels = np.array(test_ds["label"]) + 1

    pd.DataFrame({
        "text": test_df["text"].values,
        "true_label": test_labels,
        "predicted_label": test_preds,
    }).to_csv(output_dir / "test_predictions.csv", index=False)

    results = {
        "model": args.model,
        "test_metrics": test_results,
        "config": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed},
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nModel saved to {model_path}")
    print(f"Test accuracy: {test_results['eval_accuracy']:.4f}  |  Macro F1: {test_results['eval_macro_f1']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Fine-tune encoder for privacy classification")
    parser.add_argument("--model", default="jhu-clsp/ettin-encoder-150m", help="Base HuggingFace model")
    parser.add_argument("--dataset", default=DATASET_NAME, help="HuggingFace dataset id")
    parser.add_argument("--output-dir", default="outputs/privacy-model", help="Output directory")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-class-weights", dest="class_weights", action="store_false")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
