"""Fine-tune a compact instruction model with LoRA adapters."""

import argparse
import warnings
from pathlib import Path

from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from datasets.utils import logging as datasets_logging
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, TrainerCallback, set_seed)
from transformers.trainer_callback import PrinterCallback
from transformers.utils import logging as transformers_logging

from data import load_jsonl, tokenize_dataset


class TrainingProgress(TrainerCallback):
    """Display one compact progress bar for optimizer steps."""

    def __init__(self):
        self.bar = None
        self.last_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.last_step = state.global_step
        self.bar = tqdm(total=state.max_steps, desc="Fine-tuning", unit="step", dynamic_ncols=True)

    def on_step_end(self, args, state, control, **kwargs):
        if self.bar is not None:
            self.bar.update(state.global_step - self.last_step)
            self.last_step = state.global_step

    def on_train_end(self, args, state, control, **kwargs):
        if self.bar is not None:
            self.bar.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True, help="JSONL rows with instruction, optional input, and output")
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--model", default="google/flan-t5-small")
    parser.add_argument("--output-dir", default="outputs/flan-t5-lora")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-source-length", type=int, default=1024)
    parser.add_argument("--max-target-length", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--quiet", action="store_true", help="Hide checkpoint path messages")
    return parser.parse_args()


def main():
    warnings.filterwarnings("ignore")
    transformers_logging.set_verbosity_error()
    datasets_logging.disable_progress_bar()
    args = parse_args()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=args.lora_rank, lora_alpha=16,
        lora_dropout=0.05, target_modules=["q", "v"],
    ))
    if not args.quiet:
        model.print_trainable_parameters()

    train = tokenize_dataset(load_jsonl(args.train_file), tokenizer, args.max_source_length, args.max_target_length)
    valid = tokenize_dataset(load_jsonl(args.validation_file), tokenizer, args.max_source_length, args.max_target_length)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="no",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        fp16=False,
        dataloader_pin_memory=False,
        report_to="none",
        seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        processing_class=tokenizer,
    )
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(TrainingProgress())
    trainer.train()
    output = Path(args.output_dir)
    trainer.save_model(output)
    tokenizer.save_pretrained(output)
    if not args.quiet:
        print(f"Saved best adapter and tokenizer to {output}")


if __name__ == "__main__":
    main()
