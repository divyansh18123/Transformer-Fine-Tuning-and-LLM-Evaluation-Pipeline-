"""Run data preparation, LoRA fine-tuning, and paired test evaluation."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
ADAPTER_DIR = OUTPUT_DIR / "flan-t5-lora-small-run"
RESULTS_DIR = OUTPUT_DIR / "small-run-evaluation"
TRAIN_SIZE = 1000
VALIDATION_SIZE = 200
TEST_SIZE = 200
SEED = 42
MODEL = "google/flan-t5-small"
EPOCHS = 1
BATCH_SIZE = 1
MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 96


def run_stage(name: str, command: list[str]):
    print(f"\n=== {name} ===", flush=True)
    environment = os.environ.copy()
    environment["PYTHONWARNINGS"] = "ignore"
    environment["HF_HUB_VERBOSITY"] = "error"
    environment["TRANSFORMERS_VERBOSITY"] = "error"
    environment["DATASETS_VERBOSITY"] = "error"
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
    )
    if result.returncode:
        print(f"Could not complete: {name}", file=sys.stderr, flush=True)
        raise SystemExit(result.returncode)
    print(f"Completed: {name}", flush=True)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    python = sys.executable

    run_stage("1/3 Prepare dataset", [
        python, "prepare_data.py",
        "--output-dir", "data",
        "--train-size", str(TRAIN_SIZE),
        "--validation-size", str(VALIDATION_SIZE),
        "--test-size", str(TEST_SIZE),
        "--seed", str(SEED),
        "--overwrite",
        "--quiet",
    ])

    run_stage("2/3 Fine-tune model", [
        python, "train.py",
        "--train-file", str(DATA_DIR / "train.jsonl"),
        "--validation-file", str(DATA_DIR / "validation.jsonl"),
        "--model", MODEL,
        "--output-dir", str(ADAPTER_DIR),
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--gradient-accumulation", "2",
        "--max-source-length", str(MAX_SOURCE_LENGTH),
        "--max-target-length", str(MAX_TARGET_LENGTH),
        "--quiet",
    ])

    run_stage("3/3 Evaluate on held-out test data", [
        python, "evaluate.py",
        "--data-file", str(DATA_DIR / "test.jsonl"),
        "--adapter", str(ADAPTER_DIR),
        "--base-model", MODEL,
        "--output-dir", str(RESULTS_DIR),
        "--batch-size", "1",
        "--max-source-length", str(MAX_SOURCE_LENGTH),
        "--max-new-tokens", str(MAX_TARGET_LENGTH),
        "--quiet",
    ])

    metrics = json.loads((RESULTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    print("\nPipeline complete", flush=True)
    print(f"Test examples: {metrics['base_model']['examples']}")
    for name, title in (("base_model", "Base model"), ("adapter_model", "LoRA-adapted model")):
        result = metrics[name]
        print(f"{title}: ROUGE-L {result['rougeL']:.4f} | "
              f"SacreBLEU {result['sacrebleu']:.4f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.", file=sys.stderr, flush=True)
        raise SystemExit(130)
