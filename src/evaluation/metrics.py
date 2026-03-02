"""WER computation and hours-to-threshold."""

import jiwer


def compute_wer(predictions: list, references: list) -> float:
    """Word Error Rate via jiwer."""
    return jiwer.wer(references, predictions)


def hours_to_threshold(log_entries: list, metric_key: str, threshold: float) -> dict | None:
    """Find the first log entry where metric_key <= threshold.

    Args:
        log_entries: List of dicts with 'step', 'tflops', and metric_key.
        metric_key: e.g. 'wer_clean'.
        threshold: Target value.
    Returns:
        Dict with step/tflops at threshold, or None if never reached.
    """
    for entry in log_entries:
        if entry.get(metric_key, float("inf")) <= threshold:
            return {"step": entry["step"], "tflops": entry["tflops"]}
    return None
