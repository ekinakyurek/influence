import glob
import gzip
import os
import pickle
import random
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, List, Mapping, Optional, Sequence
import numpy as np
from absl import app, flags, logging
from tqdm import tqdm
from src.lama_utils import (
    abs_to_str,
    collapse_abstracts_and_scores,
    get_sentence,
)
from src.metric_utils import (
    K_EVALS,
    average_metrics,
    check_correct,
    precision_recallv2,
    reciprocal_rankv2,
)


FLAGS = flags.FLAGS


flags.DEFINE_string(
    "metrics_file",
    default=None,
    help=(
        "input metrics file that stores baseline statistics and (examples, nn"
        " abstracts)"
    ),
)

flags.DEFINE_string(
    "scores_folder", default=None, help="output file for experiment results"
)

flags.DEFINE_string(
    "output_metrics_file", default=None, help="where to output metrics"
)

flags.DEFINE_integer("seed", default=10, help="seed")

flags.DEFINE_string(
    "exp_type",
    default="layers",
    help="exp type either layers or linear combination",
)

flags.DEFINE_float(
    "alpha",
    default=0.0,
    help="Real reranker experiments, not the upper bound ones",
)

flags.DEFINE_string(
    "reweight_type", default=None, help="geometric vs arithmethic"
)

flags.DEFINE_boolean("disable_tqdm", False, help="Disable tqdm")


flags.DEFINE_string(
    "ckpt_no", default=None, help="ckpt no for single checkpoint experiments"
)

EPS = 1e-7


@dataclass
class LayerConfig:
    layer_prefixes: Sequence[str]
    layer_weights: Sequence[float] = field(default_factory=list)
    index: Optional[int] = 0

    def __post_init__(self):
        if len(self.layer_weights) == 0:
            for prefix in self.layer_prefixes:
                self.layer_weights.append(1.0)


def get_all_layer_configs(num_layers=12, exp_type="layers"):
    """Returns configurations listed below"""
    if exp_type == "layers":
        layer_configs = [
            LayerConfig(("gradients.shared",)),
            LayerConfig(("gradients.",)),
        ]

        layer_configs += [
            LayerConfig(
                (f"gradients.encoder.block.{i}", f"gradients.decoder.block.{i}")
            )
            for i in range(num_layers)
        ]

        layer_configs += [
            LayerConfig(
                (
                    "gradients.shared",
                    f"gradients.encoder.block.{i}",
                    f"gradients.decoder.block.{i}",
                )
            )
            for i in range(num_layers)
        ]

        layer_configs += [
            LayerConfig((f"gradients.encoder.block.{i}",))
            for i in range(num_layers)
        ]

        layer_configs += [
            LayerConfig(("gradients.shared", f"gradients.encoder.block.{i}"))
            for i in range(num_layers)
        ]

        layer_configs += [
            LayerConfig(
                (
                    f"activations.encoder.block.{i}",
                    f"activations.decoder.block.{i}",
                )
            )
            for i in range(num_layers + 1)
        ]

        layer_configs += [
            LayerConfig((f"activations.encoder.block.{i}",))
            for i in range(num_layers + 1)
        ]

        layer_configs += [
            LayerConfig(
                (
                    "activations.encoder.block.0",
                    "activations.decoder.block.0",
                    f"activations.encoder.block.{i}",
                    f"activations.decoder.block.{i}",
                )
            )
            for i in range(1, num_layers + 1)
        ]

        layer_configs += [
            LayerConfig(
                (
                    "activations.encoder.block.0",
                    f"activations.encoder.block.{i}",
                    f"activations.decoder.block.{i}",
                )
            )
            for i in range(1, num_layers + 1)
        ]

        layer_configs.append(LayerConfig(("activations.",)))
        layer_configs.append(LayerConfig(("activations.", "gradients.")))
        layer_configs.append(
            LayerConfig(
                (
                    "activations.encoder.block.0",
                    "activations.decoder.block.0",
                    "gradients.shared",
                )
            )
        )
        layer_configs.append(
            LayerConfig(("activations.encoder.block.0", "gradients.shared"))
        )
    else:
        layer_configs = []
        index = 0
        for a in np.linspace(-5, 5, num=10):
            for b in np.linspace(-5, 5, num=10):
                layer_configs.append(
                    LayerConfig(
                        (
                            "activations.encoder.block.0",
                            "activations.decoder.block.0",
                            "gradients.shared",
                        ),
                        [a, b, 1.0],
                        index=index,
                    )
                )
                index += 1
        for a in (0.0, 1.0):
            for b in (0.0, 1.0):
                layer_configs.append(
                    LayerConfig(
                        (
                            "activations.encoder.block.0",
                            "activations.decoder.block.0",
                            "gradients.shared",
                        ),
                        [a, b, 1.0],
                        index=index,
                    )
                )
                index += 1
    return layer_configs


def evaluate(
    example,
    abstracts,
    fact_abstracts,
    compare_fn: Callable,
    collapse=False,
):
    """Evaluate nearast abstracts to get the metrics"""
    if collapse:
        _, idxs = np.unique(
            list(map(get_sentence, fact_abstracts)), return_index=True
        )
        fact_abstracts = [fact_abstracts[id] for id in idxs]

    check_fn = partial(check_correct, compare_fn=compare_fn)

    if len(fact_abstracts) == 0:
        logging.warning(f"empty fact abstract for query: {example}")
        return None, None, None

    precision, recall = precision_recallv2(
        abstracts,
        fact_abstracts,
        check_fn,
        ks=K_EVALS,
    )
    rr = reciprocal_rankv2(
        abstracts,
        fact_abstracts,
        check_fn,
    )

    return precision, recall, rr


def rerank_with_scores(
    abstracts: Sequence[Mapping],
    layer_scores: Mapping,
    layers: Optional[LayerConfig] = None,
    collapse: bool = False,
    normalize: bool = False,
    norm_type: str = "global",
    baseline_scores: np.array = None,
    reweight_type: Optional[str] = None,
    alpha: float = 0.0,
):
    """Given layers prefixes we sum scores of these layers
       and rerank the abstracts

    Args:
        abstracts (Sequence[Mapping]): _description_
        layer_scores (Mapping): _description_
        layers (Optional[LayerConfig], optional): _description_. Defaults to None.
        collapse (bool, optional): _description_. Defaults to False.
        normalize (bool, optional): _description_. Defaults to False.
        norm_type (str, optional): _description_. Defaults to "global".
        baseline_scores (np.array, optional): _description_. Defaults to None.
        reweight_type (Optional[str], optional): _description_. Defaults to None.
        alpha (float, optional): _description_. Defaults to 0.0.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    if layers is not None:
        # Assuming our layer configurations provide prefix codes
        def findindex(key):
            indices = np.where(
                [key.startswith(layer) for layer in layers.layer_prefixes]
            )
            return indices[0]

        sum_pnames = [
            pname
            for pname in layer_scores[0].keys()
            if len(findindex(pname)) > 0
        ]

        w_weights = {
            pname: layers.layer_weights[findindex(pname)[0]]
            for pname in sum_pnames
        }
    else:
        sum_pnames = list(layer_scores[0].keys())
        w_weights = {pname: 1.0 for pname in sum_pnames}

    abstract_scores = []

    for query_scores in layer_scores:
        if normalize:
            if norm_type == "global":
                value = 0.0
                count = 0.0
                for ptype in ("activations", "gradients"):
                    sum_pnames_typed = [
                        pname for pname in sum_pnames if pname.startswith(ptype)
                    ]
                    if len(sum_pnames_typed) > 0:
                        num_typed_checkpoints = len(
                            query_scores[sum_pnames_typed[0]]
                        )
                        typed_value = 0.0

                        for i in range(num_typed_checkpoints):

                            ckpt_value = [0.0, [1e-12, 1e-12]]

                            for pname in sum_pnames_typed:
                                score = query_scores[pname][i]
                                ckpt_value[0] += score[0] * w_weights[pname]
                                ckpt_value[1][0] += score[1][0]
                                ckpt_value[1][1] += score[1][1]

                            ckpt_value = ckpt_value[0] / np.prod(
                                np.sqrt(ckpt_value[1])
                            )
                            typed_value += ckpt_value

                        typed_value /= num_typed_checkpoints
                        value += typed_value
                        count += 1
                if count != 0:
                    value /= count
                else:
                    value = 0
            elif norm_type == "local":
                value = 0.0
                for pname in sum_pnames:
                    pname_value = 0.0
                    for score in query_scores[pname]:
                        pname_value += score[0] / np.prod(np.sqrt(score[1]))
                    pname_value /= len(query_scores[pname])
                    value += pname_value * w_weights[pname]
            else:
                raise ValueError(
                    f"{norm_type} is an unknown normalization type"
                )
        else:
            value = 0.0
            for pname in sum_pnames:
                pname_value = 0.0
                for score in query_scores[pname]:
                    pname_value += score[0]
                pname_value /= len(query_scores[pname])
                value += pname_value * w_weights[pname]

        # mean over checkpoints
        abstract_scores.append(value)

    scores = np.array(abstract_scores)
    assert len(scores) == len(abstracts), f"{len(scores)} vs {len(abstracts)}"

    if reweight_type is not None and baseline_scores is not None:
        scores[scores < 0] = 0
        if reweight_type == "geometric":
            scores = np.exp(
                (1.0 - alpha) * np.log(scores + 1e-10)
                + alpha * np.log(baseline_scores + 1e-10)
            )
        else:
            scores = (1.0 - alpha) * scores + alpha * baseline_scores

    # merge abstracts and scores here
    if collapse:
        scores, abstracts = collapse_abstracts_and_scores(scores, abstracts)

    sorted_idxs = np.argsort(-scores)
    abstracts_reranked = [abstracts[i] for i in sorted_idxs]
    scores_reranked = scores[sorted_idxs]

    return abstracts_reranked, scores_reranked


def run_all_layer_configs(
    samples: List,
    scores: List,
    num_layers: int = 12,
    reweight_type: Optional[str] = None,
    alpha: float = 0.0,
    exp_type: str = "layers",
    norm_type: str = "global",
) -> Mapping:
    """Runs reranking experiments for all configurations
       listed below and returns the results

    Args:
        samples (List): List of samples
        scores (List): List of scores
        num_layers (int, optional): _description_. Defaults to 12.
        reweight_type (str, optional): _description_. Defaults to None.
        alpha (float, optional): _description_. Defaults to 0.0.
        exp_type (str, optional): _description_. Defaults to "layers".
        norm_type (str, optional): _description_. Defaults to "global".

    Returns:
        Mapping: _description_
    """
    assert len(scores) == len(samples), f"{len(scores)} vs {len(samples)}"

    logging.info(f"Processing {len(scores)} samples")

    layer_configs = get_all_layer_configs(num_layers, exp_type)

    results = {"cosine": {}, "dot": {}}

    for (index, sample) in enumerate(tqdm(samples, disable=FLAGS.disable_tqdm)):
        query = sample["example"]
        abstracts = (
            sample["nn_abstracts"]
            + sample["fact_abstracts"]
            + sample["distractors"]
        )
        rng = np.random.default_rng(0)
        rng.shuffle(abstracts)
        _, indices = np.unique(
            list(map(abs_to_str, abstracts)), return_index=True
        )
        abstracts = [abstracts[ind] for ind in indices]

        baseline_scores = []
        nn_identifiers = list(map(abs_to_str, sample["nn_abstracts"]))
        for abstract in abstracts:
            abstract_identifier = abs_to_str(abstract)
            try:
                ind = nn_identifiers.index(abstract_identifier)
                baseline_scores.append(sample["nn"]["scores"][ind])
            except ValueError:
                baseline_scores.append(0.0)
        baseline_scores = np.array(baseline_scores)

        # Get similarity scores for all individual
        # weights x {activations, gradients, both}
        score = scores[index]

        for k, result in results.items():  # cosine or dot

            if norm_type == "global" and k == "dot":
                continue

            for method in (
                "collapse",
                "full",
            ):  # eval methods

                if method == "collapse":

                    def compare_fn(a, b):
                        return get_sentence(a) == get_sentence(b)

                else:

                    def compare_fn(a, b):
                        return a["id"] == b["id"]

                    def compare_fn_relation(a, b):
                        return query["predicate_id"] in a["facts"]

                    def compare_fn_object(a, b):
                        return query["obj_uri"] in a["facts"]

                    def compare_fn_subject(a, b):
                        return query["sub_uri"] in a["facts"]

                if method not in result:
                    result[method] = {}

                is_collapse = method == "collapse"
                is_normalized = k == "cosine"

                for config in layer_configs:

                    config_name = ",".join(config.layer_prefixes)

                    if exp_type != "layers":
                        config_name = config_name + "_" + str(config.index)

                    if config_name not in result[method]:
                        result[method][config_name] = []

                    abstracts_config, scores_config = rerank_with_scores(
                        abstracts,
                        score,
                        layers=config,
                        collapse=is_collapse,
                        normalize=is_normalized,
                        norm_type=norm_type,
                        baseline_scores=baseline_scores,
                        reweight_type=reweight_type,
                        alpha=alpha,
                    )

                    precision, recall, rr = evaluate(
                        query,
                        abstracts_config,
                        sample["fact_abstracts"],
                        collapse=is_collapse,
                        compare_fn=compare_fn,
                    )

                    if precision is not None:
                        current_metrics = {
                            "index": index,
                            "precision": precision,
                            "recall": recall,
                            "rr": rr,
                            "nn_abstracts": abstracts_config[:100],
                            "nn_scores": scores_config[:100].tolist(),
                            "weights": config.layer_weights,
                        }

                        if not is_collapse:
                            for compare_fn_sub in (
                                compare_fn_relation,
                                compare_fn_object,
                                compare_fn_subject,
                            ):
                                precision_sub, recall_sub, rr_sub = evaluate(
                                    query,
                                    abstracts_config,
                                    sample["fact_abstracts"],
                                    collapse=is_collapse,
                                    compare_fn=compare_fn_sub,
                                )

                                current_metrics[
                                    "precision_" + compare_fn_sub.__name__
                                ] = precision_sub

                                current_metrics[
                                    "recall_" + compare_fn_sub.__name__
                                ] = recall_sub

                                current_metrics[
                                    "rr_" + compare_fn_sub.__name__
                                ] = rr_sub

                        result[method][config_name].append(current_metrics)

                    else:
                        logging.warning(f"metrics are none in method: {method}")

    metrics = {"cosine": {}, "dot": {}}
    for k, result in results.items():
        for (method_name, res) in result.items():
            metrics[k][method_name] = {}
            for (config_name, r) in res.items():
                average_result = average_metrics(r)
                metrics[k][method_name][config_name] = average_result
                logging.info(
                    f"{config_name}\t{method_name}\t{k}\t{average_result['mrr']}"
                )

    if norm_type == "global":
        metrics.pop("dot")

    return metrics


def main(_):
    for attr, flag_obj in FLAGS.__flags.items():
        logging.info("--%s=%s" % (attr, flag_obj.value))

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    with gzip.open(FLAGS.metrics_file, "rb") as handle:
        metrics = pickle.load(handle)

    try:
        logging.info(
            f"bm25plus mrr:  {metrics['evals']['bm25plus']['collapse']['mrr']}"
        )
    except KeyError:
        logging.info("cannot print baseline scores")

    samples = metrics["samples"]

    score_files = glob.glob(os.path.join(FLAGS.scores_folder, "*.pickle"))
    if FLAGS.ckpt_no is not None:
        score_files = [
            file
            for file in score_files
            if f"pytorch_model_{FLAGS.ckpt_no}." in file
            # if f"checkpoint-{FLAGS.ckpt_no}." in file
        ]

    logging.info(f"score files {repr(score_files)}")
    scores = None
    for score_file in score_files:
        with gzip.open(score_file, "rb") as handle:
            # queries[abstracts[params[dotproducts]]]
            ckpt_scores = pickle.load(handle)
            if scores is None:
                scores = [
                    [{} for _ in abstract_scores]
                    for abstract_scores in ckpt_scores
                ]

            for i, abstract_scores in enumerate(ckpt_scores):
                for j, score in enumerate(abstract_scores):
                    for pname, value in score.items():
                        if pname not in scores[i][j]:
                            scores[i][j][pname] = [value]
                        else:
                            scores[i][j][pname].append(value)

    global_metrics = run_all_layer_configs(
        samples,
        scores,
        exp_type=FLAGS.exp_type,
        reweight_type=FLAGS.reweight_type,
        alpha=FLAGS.alpha,
        norm_type="global",
    )

    local_metrics = run_all_layer_configs(
        samples,
        scores,
        exp_type=FLAGS.exp_type,
        reweight_type=FLAGS.reweight_type,
        alpha=FLAGS.alpha,
        norm_type="local",
    )

    for name, new_metrics in (
        ("global", global_metrics),
        ("local", local_metrics),
    ):
        if name not in metrics["evals"]:
            metrics["evals"][name] = {}
            for k, results in new_metrics.items():
                if k not in metrics["evals"][name]:
                    metrics["evals"][name][k] = {}
                for key, result in results.items():
                    metrics["evals"][name][k][key] = result

    if FLAGS.reweight_type is not None:
        FLAGS.output_metrics_file += (
            f"_{FLAGS.reweight_type}_reweight_{FLAGS.alpha}_"
        )

    logging.info(f"Logging to {FLAGS.output_metrics_file}")

    with gzip.open(FLAGS.output_metrics_file + ".pickle", "wb") as f:
        pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    app.run(main)
