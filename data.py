"""Shared dataset loading and prompt formatting for instruction tuning."""

import json
from pathlib import Path
from typing import Any

from datasets import Dataset


def format_prompt(row: dict[str, Any], variant: str = "standard") -> str:
    instruction = str(row["instruction"]).strip()
    context = str(row.get("input", "") or "").strip()
    if variant == "reworded":
        if context:
            return f"Please complete this task.\nTask: {instruction}\nAdditional context: {context}\nAnswer:"
        return f"Please complete this task.\nTask: {instruction}\nAnswer:"
    if context:
        return f"Instruction: {instruction}\nInput: {context}\nResponse:"
    return f"Instruction: {instruction}\nResponse:"


def load_jsonl(path: str | Path) -> Dataset:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object on {path}:{line_number}")
            if not str(row.get("instruction", "")).strip() or not str(row.get("output", "")).strip():
                raise ValueError(f"Each row needs non-empty 'instruction' and 'output' ({path}:{line_number})")
            row["input"] = str(row.get("input", "") or "")
            row["instruction"] = str(row["instruction"])
            row["output"] = str(row["output"])
            rows.append(row)
    if not rows:
        raise ValueError(f"Dataset contains no examples: {path}")
    return Dataset.from_list(rows)


def tokenize_dataset(dataset, tokenizer, max_source_length: int, max_target_length: int):
    def tokenize(batch):
        prompts = [
            format_prompt({"instruction": instruction, "input": context})
            for instruction, context in zip(batch["instruction"], batch["input"])
        ]
        encoded = tokenizer(prompts, max_length=max_source_length, truncation=True)
        labels = tokenizer(text_target=batch["output"], max_length=max_target_length, truncation=True)
        encoded["labels"] = labels["input_ids"]
        return encoded

    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
