"""
Task 1: News Topic Classifier Using BERT
Objective: Fine-tune a transformer model (BERT) to classify news headlines into topic categories.
Dataset: AG News Dataset (via Hugging Face Datasets)
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import evaluate

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODEL_NAME   = "bert-base-uncased"
DATASET_NAME = "ag_news"
MAX_LENGTH   = 128
BATCH_SIZE   = 16
EPOCHS       = 3
LR           = 2e-5
OUTPUT_DIR   = "./bert_news_classifier"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_NAMES  = ["World", "Sports", "Business", "Sci/Tech"]

print(f"🖥️  Using device: {DEVICE}")


# ──────────────────────────────────────────────
# 1. LOAD DATASET
# ──────────────────────────────────────────────
def load_data():
    print("\n📥 Loading AG News dataset from Hugging Face...")
    dataset = load_dataset(DATASET_NAME)
    print(f"   Train size : {len(dataset['train'])}")
    print(f"   Test size  : {len(dataset['test'])}")
    print(f"   Labels     : {LABEL_NAMES}")
    return dataset


# ──────────────────────────────────────────────
# 2. TOKENIZE
# ──────────────────────────────────────────────
def tokenize_dataset(dataset):
    print(f"\n🔤 Tokenizing with {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,   # DataCollator handles padding
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    return tokenized, tokenizer


# ──────────────────────────────────────────────
# 3. METRICS
# ──────────────────────────────────────────────
accuracy_metric = evaluate.load("accuracy")
f1_metric       = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1  = f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
    return {"accuracy": acc, "f1": f1}


# ──────────────────────────────────────────────
# 4. TRAIN
# ──────────────────────────────────────────────
def train_model(tokenized_dataset, tokenizer):
    print(f"\n🚀 Loading {MODEL_NAME} for sequence classification ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_NAMES),
        id2label={i: l for i, l in enumerate(LABEL_NAMES)},
        label2id={l: i for i, l in enumerate(LABEL_NAMES)},
    )

    # Use a small subset for faster demo (remove slice for full training)
    train_ds = tokenized_dataset["train"].shuffle(seed=42).select(range(5000))
    eval_ds  = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir              = OUTPUT_DIR,
        num_train_epochs        = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate           = LR,
        weight_decay            = 0.01,
        evaluation_strategy     = "epoch",
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        metric_for_best_model   = "f1",
        logging_steps           = 50,
        fp16                    = torch.cuda.is_available(),
        report_to               = "none",
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = eval_ds,
        tokenizer       = tokenizer,
        data_collator   = data_collator,
        compute_metrics = compute_metrics,
    )

    print(f"\n🏋️  Fine-tuning BERT on AG News (subset: 5k train / 1k eval) ...")
    trainer.train()
    return trainer, model


# ──────────────────────────────────────────────
# 5. EVALUATE
# ──────────────────────────────────────────────
def evaluate_model(trainer, tokenized_dataset):
    print("\n\n📊 Final Evaluation on test set ...")
    results = trainer.evaluate(tokenized_dataset["test"].select(range(1000)))
    print(f"   Accuracy : {results['eval_accuracy']:.4f}")
    print(f"   F1 Score : {results['eval_f1']:.4f}")
    return results


# ──────────────────────────────────────────────
# 6. PREDICT
# ──────────────────────────────────────────────
def predict(texts: list, model, tokenizer) -> list:
    """Predict category for a list of news headlines."""
    model.eval()
    inputs = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    model.to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits

    preds = torch.argmax(logits, dim=-1).cpu().numpy()
    return [LABEL_NAMES[p] for p in preds]


# ──────────────────────────────────────────────
# 7. DEPLOY (Gradio)
# ──────────────────────────────────────────────
def launch_demo(model, tokenizer):
    """Launch Gradio demo for live interaction."""
    try:
        import gradio as gr

        def classify(headline):
            pred = predict([headline], model, tokenizer)[0]
            return pred

        demo = gr.Interface(
            fn          = classify,
            inputs      = gr.Textbox(label="News Headline", placeholder="Enter a news headline..."),
            outputs     = gr.Label(label="Predicted Category"),
            title       = "📰 News Topic Classifier (BERT)",
            description = "Fine-tuned BERT on AG News — classifies headlines into World, Sports, Business, Sci/Tech",
            examples    = [
                ["Scientists discover new exoplanet with potential for life"],
                ["Stock market hits record high amid strong earnings reports"],
                ["Team wins championship in overtime thriller"],
                ["UN holds emergency meeting over escalating conflict"],
            ],
        )
        print("\n🌐 Launching Gradio demo...")
        demo.launch(share=True)

    except ImportError:
        print("\n⚠️  Gradio not installed. Run: pip install gradio")
        print("   Then call launch_demo() manually.")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Task 1: News Topic Classifier Using BERT")
    print("=" * 60)

    # Load data
    dataset = load_data()

    # Tokenize
    tokenized_dataset, tokenizer = tokenize_dataset(dataset)

    # Train
    trainer, model = train_model(tokenized_dataset, tokenizer)

    # Evaluate
    evaluate_model(trainer, tokenized_dataset)

    # Save model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n💾 Model saved to: {OUTPUT_DIR}/")

    # Demo predictions
    test_headlines = [
        "NASA launches new Mars mission with advanced rover",
        "Federal Reserve raises interest rates by 25 basis points",
        "Manchester United wins Premier League title",
        "Floods devastate coastal communities in Southeast Asia",
    ]
    print("\n\n🎯 Sample Predictions:")
    predictions = predict(test_headlines, model, tokenizer)
    for headline, pred in zip(test_headlines, predictions):
        print(f"   [{pred:10s}] {headline}")

    print("\n✅ Task 1 Complete!")
    print("\n💡 To launch interactive demo run:")
    print("   from news_classifier import launch_demo, model, tokenizer")
    print("   launch_demo(model, tokenizer)")

    # Uncomment to launch Gradio:
    # launch_demo(model, tokenizer)


if __name__ == "__main__":
    main()
