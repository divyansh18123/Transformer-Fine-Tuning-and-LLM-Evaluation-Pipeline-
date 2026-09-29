"""Download a small BillSum subset and write the project's JSONL split files."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


DATASET_ID = "FiscalNote/billsum"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--train-size", type=int, default=1500,
                        help="Number of examples for training (default: 1500)")
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing train/validation/test files")
    parser.add_argument("--quiet", action="store_true", help="Hide informational messages")
    return parser.parse_args()


def to_project_rows(split):
    return [
        {"instruction": "Summarize the following US legislation.",
         "input": row["text"], "output": row["summary"]}
        for row in split
    ]


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in tqdm(rows, desc=f"Preparing {path.stem}", unit="row"):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    if args.quiet:
        import warnings
        warnings.filterwarnings("ignore")
    if min(args.train_size, args.validation_size, args.test_size) <= 0:
        raise ValueError("Train, validation, and test sizes must all be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {name: output_dir / f"{name}.jsonl"
                   for name in ("train", "validation", "test")}
    metadata_path = output_dir / "dataset_info.json"
    existing = [path for path in [*split_paths.values(), metadata_path] if path.exists()]
    if existing and not args.overwrite:
        files = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to replace existing data files: {files}. Pass --overwrite to replace them.")

    if not args.quiet:
        print(f"Loading {DATASET_ID} from Hugging Face (first run downloads the dataset)...")
    source_train = load_dataset(DATASET_ID, split="train")
    source_test = load_dataset(DATASET_ID, split="test")
    required_train = args.train_size + args.validation_size
    if required_train > len(source_train) or args.test_size > len(source_test):
        raise ValueError(
            f"Requested {required_train} train/validation and {args.test_size} test rows, "
            f"but source splits contain {len(source_train)} and {len(source_test)} rows."
        )

    # Validation comes from the official training split; final test examples
    # come only from BillSum's separate official US test split.
    sampled_train = source_train.shuffle(seed=args.seed).select(range(required_train))
    sampled_test = source_test.shuffle(seed=args.seed).select(range(args.test_size))
    train_rows = to_project_rows(sampled_train.select(range(args.train_size)))
    validation_rows = to_project_rows(
        sampled_train.select(range(args.train_size, required_train))
    )
    test_rows = to_project_rows(sampled_test)

    write_jsonl(split_paths["train"], train_rows)
    write_jsonl(split_paths["validation"], validation_rows)
    write_jsonl(split_paths["test"], test_rows)

    metadata = {
        "dataset": DATASET_ID,
        "dataset_url": "https://huggingface.co/datasets/FiscalNote/billsum",
        "license": "CC0-1.0 (as listed on the Hugging Face dataset card)",
        "seed": args.seed,
        "source_splits": {"train": "train", "validation": "train", "test": "test"},
        "written_examples": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
        },
        "citation": "Kornilova and Eidelman (2019), BillSum: A Corpus for Automatic Summarization of US Legislation, https://aclanthology.org/D19-5406/",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    if not args.quiet:
        print(f"Wrote {len(train_rows)} train, {len(validation_rows)} validation, "
              f"and {len(test_rows)} test examples to {output_dir}")


if __name__ == "__main__":
    main()
