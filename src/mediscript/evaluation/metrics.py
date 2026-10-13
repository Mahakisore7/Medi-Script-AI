from rapidfuzz.distance import Levenshtein


def character_error_rate(pred: str, truth: str) -> float:
    if not truth:
        return 0.0 if not pred else 1.0
    return Levenshtein.distance(pred, truth) / len(truth)


def word_error_rate(pred: str, truth: str) -> float:
    pred_words, truth_words = pred.split(), truth.split()
    if not truth_words:
        return 0.0 if not pred_words else 1.0
    return Levenshtein.distance(pred_words, truth_words) / len(truth_words)


def exact_match(pred: str, truth: str) -> bool:
    return pred.strip().lower() == truth.strip().lower()


def evaluate_batch(predictions: list[str], ground_truths: list[str]) -> dict:
    cers = [character_error_rate(p, t) for p, t in zip(predictions, ground_truths, strict=False)]
    wers = [word_error_rate(p, t) for p, t in zip(predictions, ground_truths, strict=False)]
    ems = [exact_match(p, t) for p, t in zip(predictions, ground_truths, strict=False)]
    n = len(ground_truths)

    if n == 0:
        return {"cer": 0.0, "wer": 0.0, "exact_match_accuracy": 0.0, "n_samples": 0}

    return {
        "cer": sum(cers) / n,
        "wer": sum(wers) / n,
        "exact_match_accuracy": sum(ems) / n,
        "n_samples": n,
    }
