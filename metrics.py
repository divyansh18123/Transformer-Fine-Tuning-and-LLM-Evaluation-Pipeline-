"""Text generation metrics used by the evaluation CLI."""

import sacrebleu
from rouge_score import rouge_scorer


def score_texts(predictions: list[str], references: list[str]) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = [scorer.score(reference, prediction)["rougeL"].fmeasure
               for prediction, reference in zip(predictions, references)]
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return {
        "examples": len(references),
        "rougeL": sum(rouge_l) / max(len(rouge_l), 1),
        "sacrebleu": bleu.score,
    }
