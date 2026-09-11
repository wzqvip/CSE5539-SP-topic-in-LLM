"""Train head-only and LoRA ModernBERT classifiers on GLUE SST-2."""

from __future__ import annotations

import argparse
import json
import random
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("head", "lora", "both"), default="both")
    parser.add_argument("--model-name", default="answerdotai/ModernBERT-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=1)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=5539)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/modernbert-sst2"))
    return parser.parse_args()


def limit_split(dataset, limit: int | None):
    if limit is None or limit >= len(dataset):
        return dataset
    return dataset.select(range(limit))


def prepare_data(tokenizer, args: argparse.Namespace) -> tuple[DatasetDict, DataCollatorWithPadding]:
    dataset = load_dataset("nyu-mll/glue", "sst2")

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=args.max_length)

    tokenized = dataset.map(tokenize, batched=True, desc="Tokenizing SST-2")
    tokenized = tokenized.remove_columns(["sentence", "idx"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    tokenized["train"] = limit_split(tokenized["train"], args.max_train_samples)
    tokenized["validation"] = limit_split(tokenized["validation"], args.max_eval_samples)
    if "test" in tokenized:
        tokenized["test"] = limit_split(tokenized["test"], args.max_eval_samples)
    return tokenized, DataCollatorWithPadding(tokenizer=tokenizer)


def build_model(mode: str, args: argparse.Namespace):
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2, id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
    )
    if mode == "head":
        for parameter in model.base_model.parameters():
            parameter.requires_grad = False
    else:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            target_modules=["Wqkv", "Wo", "Wi"],
            modules_to_save=["classifier"],
        )
        model = get_peft_model(model, lora_config)
    return model


def count_parameters(model) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device) -> float | None:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        batch = move_batch(batch, device)
        labels = batch.get("labels")
        if labels is None or torch.any(labels < 0):
            return None
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.numel()
    return correct / total if total else None


def make_loader(dataset, collator, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


def train_one(mode: str, dataset, collator, args, device, output_dir):
    model = build_model(mode, args).to(device)
    parameters = count_parameters(model)
    train_loader = make_loader(dataset["train"], collator, args.batch_size, True)
    dev_loader = make_loader(dataset["validation"], collator, args.batch_size, False)
    test_loader = make_loader(dataset["test"], collator, args.batch_size, False)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    history = {"train_accuracy": [], "validation_accuracy": [], "test_accuracy": None}
    best_accuracy = -1.0
    best_epoch = 0
    best_state = None
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = move_batch(batch, device)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        train_accuracy = evaluate(model, train_loader, device)
        validation_accuracy = evaluate(model, dev_loader, device)
        history["train_accuracy"].append(train_accuracy)
        history["validation_accuracy"].append(validation_accuracy)
        print(f"{mode} epoch {epoch}: train={train_accuracy:.4f}, validation={validation_accuracy:.4f}")
        if validation_accuracy is not None and validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            model.save_pretrained(mode_dir / "best")
            with open(mode_dir / "best_epoch.json", "w", encoding="utf-8") as file:
                json.dump({"epoch": epoch, "validation_accuracy": validation_accuracy}, file, indent=2)

    if best_state is not None:
        model.load_state_dict(best_state)
    history["test_accuracy"] = evaluate(model, test_loader, device)
    history["best_epoch"] = best_epoch
    history["best_validation_accuracy"] = best_accuracy if best_accuracy >= 0 else None
    with open(mode_dir / "history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    return history, parameters


def save_summary(results, parameter_counts, output_dir: Path) -> None:
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    with open(output_dir / "parameter_counts.json", "w", encoding="utf-8") as file:
        json.dump(parameter_counts, file, indent=2)
    rows = ["| Approach | Accuracy (validation) |", "|---|---:|"]
    for mode, label in (("head", "Head tuning"), ("lora", "LoRA")):
        if mode in results:
            rows.append(f"| {label} | {results[mode]['best_validation_accuracy']:.4f} |")
    (output_dir / "table1.md").write_text("\n".join(rows) + "\n", encoding="utf-8")

    plt.figure(figsize=(8, 5))
    for mode, label in (("head", "Head tuning"), ("lora", "LoRA")):
        if mode not in results:
            continue
        epochs = range(1, len(results[mode]["train_accuracy"]) + 1)
        plt.plot(epochs, results[mode]["train_accuracy"], marker="o", label=f"{label} train")
        plt.plot(epochs, results[mode]["validation_accuracy"], marker="o", linestyle="--", label=f"{label} dev")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curves.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset, collator = prepare_data(tokenizer, args)
    modes = ("head", "lora") if args.mode == "both" else (args.mode,)
    results = {}
    parameter_counts = {}
    for mode in modes:
        set_seed(args.seed)
        results[mode], parameter_counts[mode] = train_one(
            mode, dataset, collator, args, device, args.output_dir
        )
    save_summary(results, parameter_counts, args.output_dir)
    print(f"Wrote experiment artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()