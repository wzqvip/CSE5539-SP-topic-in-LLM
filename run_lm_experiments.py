"""Run the homework's language-model perplexity and sampling experiments."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_TEXT = (
    "Language models learn statistical patterns in text. "
    "A well-trained model assigns higher probability to coherent word sequences. "
    "Perplexity measures how surprised the model is by a document."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="distilgpt2")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--seed", type=int, default=5539)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/language-model"))
    return parser.parse_args()


def perplexity(model, tokenizer, text: str, device: torch.device) -> float:
    encoded = tokenizer(text, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        loss = model(**encoded, labels=encoded["input_ids"]).loss
    return float(torch.exp(loss).cpu())


def shuffled_text(text: str, seed: int) -> str:
    words = text.split()
    random.Random(seed).shuffle(words)
    return " ".join(words)


def generate(model, tokenizer, prompt: str, temperature: float, max_new_tokens: int, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        if temperature == 0:
            output = model.generate(
                **inputs, do_sample=False, max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            output = model.generate(
                **inputs, do_sample=True, temperature=temperature,
                top_k=0, max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    original = args.text
    shuffled = shuffled_text(original, args.seed)
    results = {
        "model": args.model_name,
        "device": str(device),
        "seed": args.seed,
        "text": original,
        "shuffled_text": shuffled,
        "perplexity": {
            "original": perplexity(model, tokenizer, original, device),
            "shuffled": perplexity(model, tokenizer, shuffled, device),
        },
        "samples": {},
    }
    for temperature in (0, 0.3, 0.6, 0.9, 1.2, 1.5):
        results["samples"][str(temperature)] = generate(
            model, tokenizer, "Once upon a time", temperature,
            args.max_new_tokens, device,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "samples.md").open("w", encoding="utf-8") as file:
        file.write("# DistilGPT-2 sampling outputs\n\n")
        for temperature, sample in results["samples"].items():
            file.write(f"## Temperature {temperature}\n\n{sample}\n\n")
    print(json.dumps(results["perplexity"], indent=2))
    print(f"Wrote language-model artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()