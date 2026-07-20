"""
Ablation study: 4 fine-tuning strategies for Whisper-based speech emotion recognition.

Usage:
    python train.py
    python train.py --smoke      # 1-epoch sanity check
    python train.py --only lora_q_v_mlp   # run a single config
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from transformers import (EarlyStoppingCallback, Trainer, TrainingArguments,
                          WhisperFeatureExtractor)

from data import (EMOTIONS, build_dataset, download_ravdess, get_class_weights,
                  preprocess_audio, speaker_disjoint_split)
from model import MODEL_ID, build_model

DATA_DIR = Path("./ravdess_data")
RESULTS_DIR = Path("./results")
SEED = 42

EXPERIMENTS = [
    {"name": "frozen_baseline", "method": "frozen", "target_modules": None,
     "lr": 1e-3},
    {"name": "lora_q_v",        "method": "lora",
     "target_modules": ["q_proj", "v_proj"],                       "lr": 1e-4},
    {"name": "lora_q_k_v_out",  "method": "lora",
     "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"], "lr": 1e-4},
    {"name": "lora_q_v_mlp",    "method": "lora",
     "target_modules": ["q_proj", "v_proj", "fc1", "fc2"],         "lr": 1e-4},
]


def collate(features):
    batch = {
        "input_features": torch.tensor(
            np.stack([f["input_features"] for f in features]), dtype=torch.float32),
        "labels": torch.tensor([f["label"] for f in features], dtype=torch.long),
    }
    if "attention_mask" in features[0]:
        batch["attention_mask"] = torch.tensor(
            np.stack([f["attention_mask"] for f in features]), dtype=torch.long)
    return batch


def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=-1)
    return {
        "accuracy":    accuracy_score(pred.label_ids, preds),
        "f1_weighted": f1_score(pred.label_ids, preds, average="weighted"),
        "f1_macro":    f1_score(pred.label_ids, preds, average="macro"),
    }


def run_experiment(cfg, train_ds, eval_ds, class_weights, smoke=False):
    """Train one model and return its metrics + predictions on eval."""
    print("\n" + "=" * 70)
    print(f"  {cfg['name']}")
    print("=" * 70)

    torch.manual_seed(SEED)
    model, n_train, n_total = build_model(
        cfg["method"], num_labels=len(EMOTIONS),
        target_modules=cfg["target_modules"],
        class_weights=class_weights,
    )

    args = TrainingArguments(
        output_dir=str(RESULTS_DIR / cfg["name"]),
        num_train_epochs=1 if smoke else 10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=cfg["lr"],
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        logging_steps=20,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to="none",
        label_names=["labels"],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collate, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    preds = trainer.predict(eval_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)

    # Save best checkpoint (LoRA adapter or full state)
    out_dir = RESULTS_DIR / cfg["name"] / "best"
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg["method"] == "lora":
        model.encoder.save_pretrained(out_dir / "lora_adapter")
    torch.save(model.head.state_dict(), out_dir / "head.pt")

    result = {
        "name": cfg["name"],
        "method": cfg["method"],
        "target_modules": ",".join(cfg["target_modules"]) if cfg["target_modules"] else "—",
        "trainable_params": int(n_train),
        "trainable_pct":    round(100 * n_train / n_total, 2),
        "accuracy":    float(preds.metrics["test_accuracy"]),
        "f1_weighted": float(preds.metrics["test_f1_weighted"]),
        "f1_macro":    float(preds.metrics["test_f1_macro"]),
    }

    del trainer, model
    torch.cuda.empty_cache()
    return result, y_pred


def plot_confusion_matrix(cm, title, save_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(EMOTIONS))); ax.set_yticks(range(len(EMOTIONS)))
    ax.set_xticklabels(EMOTIONS, rotation=45, ha="right"); ax.set_yticklabels(EMOTIONS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() * 0.6
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.colorbar(im); plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_results_table(df, save_path):
    md = ("| Method | Target modules | Trainable params | Accuracy | "
          "Macro F1 | Weighted F1 |\n|---|---|---|---|---|---|\n")
    for _, r in df.iterrows():
        params = f"{r['trainable_params']/1e6:.2f}M ({r['trainable_pct']}%)"
        md += (f"| {r['name']} | `{r['target_modules']}` | {params} | "
               f"{r['accuracy']*100:.1f}% | {r['f1_macro']:.3f} | "
               f"{r['f1_weighted']:.3f} |\n")
    Path(save_path).write_text(md)


def main(smoke=False, only=None):
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    print("CUDA:", torch.cuda.is_available())

    # Data
    audio_dir = download_ravdess(DATA_DIR)
    ds = build_dataset(audio_dir)
    train_ds, eval_ds = speaker_disjoint_split(ds)

    if smoke:
        train_ds = train_ds.shuffle(seed=SEED).select(range(32))
        eval_ds  = eval_ds.shuffle(seed=SEED).select(range(16))

    print(f"Train: {len(train_ds)}  |  Eval: {len(eval_ds)}")

    # Preprocess
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
    train_ds = preprocess_audio(train_ds, feature_extractor)
    eval_ds  = preprocess_audio(eval_ds,  feature_extractor)
    feature_extractor.save_pretrained(RESULTS_DIR / "feature_extractor")

    class_weights = get_class_weights(train_ds)

    # Run experiments
    experiments = [e for e in EXPERIMENTS if (only is None or e["name"] == only)]
    results, predictions = [], {}
    for exp in experiments:
        r, y_pred = run_experiment(exp, train_ds, eval_ds, class_weights, smoke=smoke)
        results.append(r)
        predictions[exp["name"]] = y_pred

    # Save comparison table
    df = pd.DataFrame(results).sort_values("f1_weighted", ascending=False)
    df.to_csv(RESULTS_DIR / "results.csv", index=False)
    write_results_table(df, RESULTS_DIR / "results_table.md")
    print("\n" + df[["name", "trainable_pct", "accuracy",
                     "f1_weighted", "f1_macro"]].to_string(index=False))

    # Confusion matrix + classification report for best run
    best = df.iloc[0]
    print(f"\nBest run: {best['name']} (F1 weighted = {best['f1_weighted']:.3f})")

    y_true = np.array(eval_ds["label"])
    y_pred = predictions[best["name"]]
    labels = np.arange(len(EMOTIONS))   
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plot_confusion_matrix(cm, f"Confusion matrix — {best['name']}",
                          RESULTS_DIR / "confusion_matrix.png")
    print(classification_report(y_true, y_pred, labels=labels,
                                target_names=EMOTIONS, digits=3, zero_division=0))

    # Persist all metrics
    (RESULTS_DIR / "metrics.json").write_text(json.dumps({
        "results": results,
        "best": best["name"],
        "labels": EMOTIONS,
        "confusion_matrix_best": cm.tolist(),
    }, indent=2, default=float))

    print(f"\n✓ All artifacts saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="1-epoch sanity check")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only one experiment by name")
    args = parser.parse_args()
    main(smoke=args.smoke, only=args.only)
