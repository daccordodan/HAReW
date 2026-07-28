"""Benchmark comparison harness against prior-art HAR algorithms.

Source: Paper 2, Sec. 6.6, Figs. 9b-9c.

Benchmarks: DeepSense [11], EI [12], MatNet-eCSI [13] -- all retrained on
the same S1 training portion as SHARP, for a fair comparison. Paper 2 shows
these amplitude/phase-based (non-Doppler) methods perform comparably to
SHARP only on S1, and substantially degrade on S2-S7 (even same-environment/
different-day cases) -- this module exists specifically to reproduce that
generalization gap, not just absolute accuracy numbers.
"""

from __future__ import annotations

BENCHMARK_MODELS = ("DeepSense", "EI", "MatNet-eCSI", "SHARP")


def train_benchmark_model(model_name: str, train_dataset) -> object:
    """Trains one of the benchmark models on the S1 training split.

    Args:
        model_name: One of BENCHMARK_MODELS.
        train_dataset: S1 training dataset (see DopplerTraceDataset in
            src/data/doppler_trace_dataset.py).

    Returns:
        Trained model object (type depends on which benchmark is implemented).
    """
    raise NotImplementedError("TODO: implement/port DeepSense, EI, MatNet-eCSI architectures.")


def compare_generalization_gap(results_by_model_and_set: dict[str, dict[str, float]]) -> dict:
    """Quantifies the S1-to-S2..S7 accuracy drop for each benchmark model.

    Args:
        results_by_model_and_set: Nested dict {model_name: {set_id: accuracy}}.

    Returns:
        Dict {model_name: generalization_gap}, where generalization_gap is the
        difference between S1 accuracy and mean accuracy across S2-S7. Used
        to reproduce the qualitative finding of Paper 2 Figs. 9b-9c: SHARP's
        gap should be much smaller than the other benchmarks' gaps.
    """
    raise NotImplementedError("TODO: implement gap computation and comparison table.")
