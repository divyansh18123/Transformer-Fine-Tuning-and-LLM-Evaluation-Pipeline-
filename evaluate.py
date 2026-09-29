"""Evaluate the base model and its LoRA adapter on the same held-out JSONL set."""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from data import format_prompt, load_jsonl
from metrics import score_texts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--adapter", required=True, help="Directory saved by train.py")
    parser.add_argument("--base-model", default="google/flan-t5-small")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-source-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--prompt-sensitivity", action="store_true",
                        help="Also evaluate an equivalent reworded prompt template")
    parser.add_argument("--quiet", action="store_true", help="Do not print the final metrics JSON")
    args = parser.parse_args()

    rows = load_jsonl(args.data_file)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    examples = [rows[i] for i in range(len(rows))]

    def generate(model, variant, model_label):
        generated_text = []
        starts = range(0, len(examples), args.batch_size)
        for start in tqdm(starts, desc=f"Generating {model_label}", unit="batch"):
            batch = examples[start:start + args.batch_size]
            inputs = tokenizer([format_prompt(row, variant) for row in batch], padding=True, truncation=True,
                               max_length=args.max_source_length, return_tensors="pt").to(device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
            generated_text.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        return generated_text

    references = [row["output"] for row in examples]
    # Evaluate the unmodified base model first, then load the trained adapter
    # onto the same base weights and evaluate it with identical generation settings.
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model).to(device)
    base_model.eval()
    base_predictions = generate(base_model, "standard", "base model")
    base_alternate = generate(base_model, "reworded", "base model, reworded prompt") if args.prompt_sensitivity else None
    del base_model

    adapted_model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    adapted_model = PeftModel.from_pretrained(adapted_model, args.adapter).to(device)
    adapted_model.eval()
    adapter_predictions = generate(adapted_model, "standard", "LoRA-adapted model")
    adapter_alternate = generate(adapted_model, "reworded", "LoRA-adapted model, reworded prompt") if args.prompt_sensitivity else None
    del adapted_model

    results = {
        "base_model": score_texts(base_predictions, references),
        "adapter_model": score_texts(adapter_predictions, references),
    }
    if args.prompt_sensitivity:
        results["base_model"]["reworded_prompt"] = score_texts(base_alternate, references)
        results["base_model"]["prompt_output_change_rate"] = sum(
            a.strip().casefold() != b.strip().casefold()
            for a, b in zip(base_predictions, base_alternate)
        ) / max(len(base_predictions), 1)
        results["adapter_model"]["reworded_prompt"] = score_texts(adapter_alternate, references)
        results["adapter_model"]["prompt_output_change_rate"] = sum(
            a.strip().casefold() != b.strip().casefold()
            for a, b in zip(adapter_predictions, adapter_alternate)
        ) / max(len(adapter_predictions), 1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(examples):
            record = {
                "instruction": row["instruction"], "input": row.get("input", ""),
                "reference": row["output"],
                "base_prediction": base_predictions[index],
                "adapter_prediction": adapter_predictions[index],
                "human_review": "",
            }
            if args.prompt_sensitivity:
                record["base_reworded_prediction"] = base_alternate[index]
                record["adapter_reworded_prediction"] = adapter_alternate[index]
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    if not args.quiet:
        print(json.dumps(results, indent=2))
    if not args.quiet:
        print(f"Saved base and adapter outputs for review to {output_dir / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
