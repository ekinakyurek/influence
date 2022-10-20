from typing import Any, Callable, List, Mapping, Sequence, Tuple
import numpy as np


K_EVALS: Tuple[int, ...] = (1, 3, 5, 10, 25)


def precision_recall(
    nearest_ids: List,
    correct_ids: List,
    ks: Tuple[int, ...] = K_EVALS,
) -> Tuple[Mapping[int, float], Mapping[int, float]]:
    precision, recall = {}, {}
    for k in ks:
        nn_k = nearest_ids[:k]
        precision[k] = len([1 for id in nn_k if id in correct_ids]) / k
        recall[k] = len([1 for id in correct_ids if id in nn_k]) / len(
            correct_ids
        )
    return precision, recall


def reciprocal_rank(nearest_ids: List[int], correct_ids: List[int]) -> int:
    for i, id in enumerate(nearest_ids):
        if id in correct_ids:
            return 1 / (i + 1)
    return 0


def precision_recallv2(
    abstracts: List[Mapping[str, Any]],
    fact_abstracts: List[Mapping[str, Any]],
    check_correct: Callable,
    ks: Tuple[int, ...] = K_EVALS,
) -> Tuple[Mapping[int, float], Mapping[int, float]]:
    """Calculate precision and recall given nearest ids and correct ids"""
    precision, recall = {}, {}
    for k in ks:
        nn_k = abstracts[:k]
        precision[k] = (
            len([1 for a in nn_k if check_correct(a, fact_abstracts)]) / k
        )
        recall[k] = len(
            [1 for a in fact_abstracts if check_correct(a, nn_k)]
        ) / len(fact_abstracts)
    return precision, recall


def reciprocal_rankv2(
    abstracts: List[Mapping[str, Any]],
    fact_abstracts: List[Mapping[str, Any]],
    check_correct: Callable,
) -> int:
    """Return reciprocal rank score"""
    for i, a in enumerate(abstracts):
        if check_correct(a, fact_abstracts):
            return 1 / (i + 1)
    return 0


def average_metrics(results: List[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Average the metrics over samples"""
    metrics = {
        key: {}
        for key in results[0].keys()
        if key.startswith("precision")
        or key.startswith("recall")
        or key.startswith("rr")
    }

    keys = list(metrics.keys())
    for k in K_EVALS:
        for key in keys:
            if k != 1 and key.startswith("rr"):
                pass
            elif k == 1 and key.startswith("rr"):
                metrics["m" + key] = np.mean([res[key] for res in results])
            else:
                metrics[key][k] = np.mean([res[key][k] for res in results])

    metrics["samples"] = results
    return metrics


def check_correct(
    a1: Mapping,
    fact_abstracts: Sequence[Mapping],
    compare_fn: Callable = lambda x: x["sentence_uris"],
) -> bool:
    """Check if a1 is in fact_abstracts

    Args:
        a1 (Mapping): Retrieived abstract
        fact_abstracts (Sequence[Mapping]): Candidate facts
        collapse (bool): Whether it is collapsed evaluation for abstracts

    Returns:
        bool: True if the abstract is in fact_abstracts else False
    """
    return any((compare_fn(a1, a) for a in fact_abstracts))
