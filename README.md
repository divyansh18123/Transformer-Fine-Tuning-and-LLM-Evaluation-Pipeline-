# Transformer Fine-Tuning and Evaluation Pipeline
LoRA fine-tuning and evaluation pipeline for summarization with Flan-T5, PyTorch, and Hugging Face. Includes BillSum data preparation and base-versus-adapter evaluation using ROUGE-L and SacreBLEU.

A small, reproducible instruction-tuning project using PyTorch, Hugging Face Transformers, and LoRA (PEFT). It covers input validation, tokenization, training with validation and best-checkpoint selection, batched generation, reference-based metrics, and per-example qualitative review. The default base model is `google/flan-t5-small`; training saves an adapter rather than a second full model.

## Files

- `train.py` — fine-tunes a sequence-to-sequence model with LoRA.
- `evaluate.py` — generates held-out predictions, writes metrics and reviewable examples; optionally measures prompt sensitivity.
- `prepare_data.py` — downloads and converts the BillSum dataset to local train, validation, and test JSONL files.
- `main.py` — runs the small data-preparation, training, and paired-evaluation pipeline in order.
- `data.py` — validates JSONL data and formats/tokenizes prompts.
- `metrics.py` — computes ROUGE-L and SacreBLEU.
- `requirements.txt` — Python dependencies.

## Setup

Use Python 3.10 or newer. Install PyTorch appropriate for your machine from the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install the remaining project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The first run downloads the selected Hugging Face model. Training can run on CPU but will be slow; a CUDA GPU is recommended. The example defaults use conservative batch sizes for modest hardware. Reduce `--batch-size` if memory is limited.

## Dataset format

Provide UTF-8 JSON Lines files. Every non-empty line must be a JSON object with a non-empty `instruction` and `output`. `input` is optional and may be an empty string. Keep train, validation, and final test examples separate; do not report metrics on training examples.

```json
{"instruction":"Summarize the passage in one sentence.","input":"A short passage goes here.","output":"A one-sentence summary goes here."}
{"instruction":"Name the capital of France.","input":"","output":"Paris."}
```

Store your own data outside version control (for example under `data/`); `.gitignore` excludes JSONL files there.

## Download the included dataset

This project uses [BillSum](https://huggingface.co/datasets/FiscalNote/billsum), a public dataset for summarizing US congressional and California bills. The Hugging Face dataset card lists its license as CC0-1.0. The US `train` and `test` splits contain 18,949 and 3,269 examples. The full dataset download is roughly 114 MB and its prepared cache uses roughly 340 MB, according to the card metadata. BillSum is focused on legislation, so results describe legal-document summarization rather than general-purpose summarization. See the [dataset card](https://huggingface.co/datasets/FiscalNote/billsum) and [BillSum paper](https://aclanthology.org/D19-5406/) before reusing the data elsewhere.

The preparation script downloads a deterministic small subset and converts it to this project's instruction/input/output format. It samples training and validation examples only from the official training split, and test examples only from the official test split:

```bash
python prepare_data.py
```

By default, this writes 1,500 training rows, 200 validation rows, and 300 held-out test rows under `data/`, plus `dataset_info.json` with the dataset source, license, split counts, seed, and citation. The first run downloads the dataset from Hugging Face. Existing split files are protected; pass `--overwrite` to replace them. Change subset sizes with `--train-size`, `--validation-size`, and `--test-size`.

BillSum documents can be long, so the normal training and evaluation defaults allow up to 1,024 source tokens; longer documents are truncated by the tokenizer. Record this limit with results.

## Run the small end-to-end pipeline

To run a quick CPU-friendly experiment with 1000 training, 200 validation, and 200 test examples, one training epoch, batch size 1, and a 256-token input limit:

```bash
python main.py
```

The script displays progress bars for data preparation, training, and generation by both models, then prints the final metrics. Warnings, raw commands, and artifact paths are hidden. It replaces the three generated JSONL split files in `data/` with the selected subset and saves results under `outputs/`. This small run checks the pipeline and provides preliminary metrics; it is not a reliable benchmark of summarization quality.

## Train

```bash
python train.py \
  --train-file data/train.jsonl \
  --validation-file data/validation.jsonl \
  --output-dir outputs/flan-t5-lora
```

Useful options include `--model`, `--epochs`, `--learning-rate`, `--batch-size`, `--gradient-accumulation`, `--max-source-length`, and `--max-target-length`. Training evaluates and saves at each epoch, then retains the best checkpoint by validation loss. The output directory contains the LoRA adapter and tokenizer needed for inference; provide the same base model when evaluating if you changed the default.

## Evaluate

Use the held-out test set created by `prepare_data.py`:

```bash
python evaluate.py \
  --data-file data/test.jsonl \
  --adapter outputs/flan-t5-lora \
  --output-dir outputs/test-results
```

The command evaluates both the unmodified base model and the LoRA-adapted model on the exact same test examples with the same generation settings. `metrics.json` contains separate ROUGE-L and SacreBLEU scores under `base_model` and `adapter_model`. `predictions.jsonl` puts both models' responses beside each reference so you can review them directly. The code produces these two result sets without calculating or asserting a winner.

To also measure sensitivity to a second, semantically equivalent prompt wording:

```bash
python evaluate.py --data-file data/test.jsonl \
  --adapter outputs/flan-t5-lora --output-dir outputs/prompt-study \
  --prompt-sensitivity
```

This adds alternate-template metrics and an output-change rate for each model, plus both alternate generations to `predictions.jsonl`. The same file includes reference, input, each model's predictions, and a blank `human_review` field. Use it to inspect instruction following and factual support. These text-overlap metrics do not detect hallucinations or judge instruction following on their own; those require task-aware checks and human review.

## Reproducibility and scope

Set `--seed` to control data/model randomness. Record the model ID, dataset version, command-line settings, and hardware alongside any reported results. This is a compact pipeline intended for small fine-tuning experiments; it does not claim production serving, safety evaluation, or reliable factuality measurement.
