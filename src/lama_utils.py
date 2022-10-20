from typing import List, Mapping
import numpy as np


def get_sentence(abstract: Mapping) -> str:
    targets = (
        abstract["targets_pretokenized"].replace("<extra_id_0> ", "").strip()
    )
    sentence = abstract["inputs_pretokenized"].replace("<extra_id_0>", targets)
    return sentence


def abs_to_str(x: Mapping) -> str:
    return x["inputs_pretokenized"] + x["targets_pretokenized"]


def collapse_abstracts_and_scores(
    scores: List[float], abstracts: List[Mapping]
):
    sentence_to_indices = {}
    for i, a in enumerate(abstracts):
        sentence = get_sentence(a)
        if sentence in sentence_to_indices:
            sentence_to_indices[sentence].append(i)
        else:
            sentence_to_indices[sentence] = [i]
    sentence_scores = []
    sentence_indices = []
    scores = np.array(scores)
    for (sentence, indices) in sentence_to_indices.items():
        i_max = np.argmax(scores[indices])
        i_max = indices[i_max]
        sentence_indices.append(i_max)
        sentence_scores.append(scores[i_max])

    return np.array(sentence_scores), [abstracts[j] for j in sentence_indices]
