import copy
import gzip
import json
import pickle
import random
from functools import partial
from typing import Callable, Mapping
import numpy as np
from absl import app, flags, logging
from tqdm import tqdm
from transformers import MT5ForConditionalGeneration, T5Tokenizer
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
    "output_metrics_prefix",
    default=None,
    help="output file for experiment results",
)

flags.DEFINE_string(
    "checkpoint_folders",
    default=None,
    help="last checkpoint of the model to evaluate",
)

flags.DEFINE_integer(
    "beam_size", default=3, help="beam size for accuracy calculations"
)

flags.DEFINE_integer("seed", default=10, help="seed")

flags.DEFINE_bool(
    "only_corrects",
    default=False,
    help="evaluate only on correctly predicted examples",
)

flags.DEFINE_bool(
    "only_wrongs",
    default=False,
    help="evaluate only on wrong predicted examples",
)

flags.DEFINE_bool(
    "only_learned", default=False, help="evaluate only learned examples"
)

flags.DEFINE_string(
    "samples_from_exp", default=None, help="exp json to read samples"
)

flags.DEFINE_integer("gpu", default=0, help="gpu to use")

flags.DEFINE_boolean("disable_tqdm", False, help="Disable tqdm")


def tokenize(tokenizer: T5Tokenizer, record: Mapping):
    """Tokenize the inputs and targets of a record"""
    inputs = tokenizer(
        record["inputs_pretokenized"],
        return_tensors="pt",
        max_length=2048,
        truncation=True,
    ).input_ids

    targets = tokenizer(
        record["targets_pretokenized"], return_tensors="pt"
    ).input_ids

    return {"inputs": inputs, "targets": targets[:, :-1]}


def evaluate(
    example, abstracts, fact_abstracts, compare_fn: Callable, collapse=False
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

    # nn_ids = [a['page_uri'] for a in abstracts]
    precision, recall = precision_recallv2(
        abstracts,
        fact_abstracts,
        check_fn,
        ks=K_EVALS,
    )

    rr = reciprocal_rankv2(abstracts, fact_abstracts, check_fn)
    return precision, recall, rr


def trim(output):
    """Trim the outputs for the accuracy evaluation."""
    output = output.replace("<extra_id_0>", "")
    index = output.find("<extra_id_1>")
    if index != -1:
        output = output[:index]
    output = output.strip()
    if len(output) > 0:
        if output[-1] == ".":
            output = output[:-1]
    return output.lower()


def run_random_baseline(samples):
    metrics = {}
    for index, sample in enumerate(samples):
        query = sample["example"]
        target = query["targets_pretokenized"]
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

        target_abstracts = [
            a for a in abstracts if a["targets_pretokenized"] == target
        ]

        other_abstracts = [
            a for a in abstracts if a["targets_pretokenized"] != target
        ]

        abstracts_reranked = target_abstracts + other_abstracts
        scores_reranked = [1 for i in range(len(target_abstracts))]
        scores_reranked += [0 for i in range(len(other_abstracts))]
        fact_abstracts = sample["fact_abstracts"]

        for method in ("collapse", "full"):

            collapse = method == "collapse"

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

            if method not in metrics:
                metrics[method] = []

            results = metrics[method]

            current_scores, current_abstracts = (
                scores_reranked,
                abstracts_reranked,
            )

            if collapse:
                (
                    current_scores,
                    current_abstracts,
                ) = collapse_abstracts_and_scores(
                    current_scores, current_abstracts
                )

            precision, recall, rr = evaluate(
                query,
                current_abstracts,
                fact_abstracts,
                collapse=collapse,
                compare_fn=compare_fn,
            )

            if precision is not None:
                current_metrics = {
                    "index": index,
                    "precision": precision,
                    "recall": recall,
                    "rr": rr,
                    "nn_abstracts": abstracts_reranked[:100],
                    "nn_scores": scores_reranked[:100],
                }
                if not collapse:
                    for compare_fn_sub in (
                        compare_fn_relation,
                        compare_fn_object,
                        compare_fn_subject,
                    ):
                        precision_sub, recall_sub, rr_sub = evaluate(
                            query,
                            current_abstracts,
                            fact_abstracts,
                            collapse=collapse,
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

                results.append(current_metrics)

    for method in metrics.keys():
        metrics[method] = average_metrics(metrics[method])
    return metrics


def rerun_baseline(samples):
    metrics = {}
    for method in ("collapse", "full"):

        if method not in metrics:
            metrics[method] = []

        collapse = method == "collapse"

        for sample_index, sample in enumerate(copy.deepcopy(samples)):
            query = sample["example"]

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

            scores, abstracts = (sample["nn"]["scores"], sample["nn_abstracts"])

            if collapse:
                scores, abstracts = collapse_abstracts_and_scores(
                    scores, abstracts
                )

            precision, recall, rr = evaluate(
                query,
                abstracts,
                sample["fact_abstracts"],
                collapse=collapse,
                compare_fn=compare_fn,
            )

            if precision is None:
                continue

            if sample["rr"] != rr:
                logging.info(
                    "original scores are changed -- probably due to a"
                    f" modification in evaluation -- method: {method}"
                )
                logging.info(f"example: {sample['example']}")
                logging.info(
                    "original ones:"
                    f" {(sample['precision'], sample['recall'], sample['rr'])}"
                )
                logging.info(f"new ones: {(precision, recall, rr)}")

            sample["precision"] = precision
            sample["recall"] = recall
            sample["rr"] = rr

            result = {
                "index": sample_index,
                "precision": precision,
                "recall": recall,
                "rr": rr,
                "nn_abstracts": samples[sample_index]["nn_abstracts"][:100],
                "nn_scores": samples[sample_index]["nn"]["scores"][:100],
            }

            if not collapse:
                for compare_fn_sub in (
                    compare_fn_relation,
                    compare_fn_object,
                    compare_fn_subject,
                ):
                    precision_sub, recall_sub, rr_sub = evaluate(
                        query,
                        abstracts,
                        sample["fact_abstracts"],
                        collapse=collapse,
                        compare_fn=compare_fn_sub,
                    )

                    result[
                        "precision_" + compare_fn_sub.__name__
                    ] = precision_sub

                    result["recall_" + compare_fn_sub.__name__] = recall_sub

                    result["rr_" + compare_fn_sub.__name__] = rr_sub

            metrics[method].append(result)

        logging.info(f"Samples filtered: {len(samples)-len(metrics[method])}")

        metrics[method] = average_metrics(metrics[method])

    return metrics


def get_model_accuracy(
    model, tokenizer: T5Tokenizer, samples, beam_size=3, topk=3
):
    """Get prediction labels for the given samples"""
    labels = []
    for k, sample in enumerate(tqdm(samples, disable=FLAGS.disable_tqdm)):
        raw_input = sample["example"]["inputs_pretokenized"]
        data = tokenize(tokenizer, sample["example"])
        inputs = data["inputs"]
        target = trim(sample["example"]["targets_pretokenized"])
        outputs = model.generate(
            input_ids=inputs.cuda(model.cuda_no),
            num_beams=beam_size,
            num_return_sequences=topk,
            max_length=20,
        )

        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs = tuple(map(trim, outputs))
        labels.append(target in outputs)
        if k < 50:
            logging.info(
                f"Inputs: {raw_input}, Target: {target}, Outputs: {outputs}"
            )
    return np.array(labels)


EPS = 1e-7


def main(_):
    for attr, flag_obj in FLAGS.__flags.items():
        logging.info("--%s=%s" % (attr, flag_obj.value))

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    with gzip.open(FLAGS.metrics_file) as handle:
        original_result = pickle.load(handle)

    tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")

    checkpoint_folders = FLAGS.checkpoint_folders.split(",")

    model = MT5ForConditionalGeneration.from_pretrained(
        checkpoint_folders[-1], local_files_only=True
    ).cuda(FLAGS.gpu)

    model.eval()
    model.cuda_no = FLAGS.gpu

    samples = original_result["samples"]

    rng = np.random.default_rng(FLAGS.seed)
    rng.shuffle(samples)

    logging.info(f"Number of samples in original: {len(samples)}")

    labels = get_model_accuracy(
        model,  # Last checkpoint is the best accuracy.
        tokenizer,
        samples,
        beam_size=FLAGS.beam_size,
        topk=1,
    )

    logging.info(f"Mean accuracy of last checkpoint is {np.mean(labels)}")

    assert not (FLAGS.only_corrects and FLAGS.only_wrongs)

    if FLAGS.samples_from_exp is not None:
        with open(FLAGS.samples_from_exp) as f:
            exp_metrics = json.load(f)
        exp_inputs = [
            sample["example"]["inputs_pretokenized"]
            for sample in exp_metrics["evals"]["bm25plus"]["samples"]
        ]
        exp_uris = set(exp_inputs)
        samples = [
            sample
            for sample in samples
            if sample["example"]["inputs_pretokenized"] in exp_uris
        ]
        original_result["samples"] = samples
    else:
        if FLAGS.only_corrects:

            model = MT5ForConditionalGeneration.from_pretrained(
                checkpoint_folders[0], local_files_only=True
            ).cuda(FLAGS.gpu)
            model.eval()
            model.cuda_no = FLAGS.gpu

            labels_zero = get_model_accuracy(
                model, tokenizer, samples, beam_size=FLAGS.beam_size
            )

            logging.info(
                f"Mean accuracy of first checkpointis {np.mean(labels_zero)}"
            )

            samples = [
                samples[i] for i in range(len(labels_zero)) if labels_zero[i]
            ]

            original_result["samples"] = samples

        elif FLAGS.only_wrongs:
            samples = [samples[i] for i in range(len(labels)) if not labels[i]]
            original_result["samples"] = samples

        elif FLAGS.only_learned:
            model = MT5ForConditionalGeneration.from_pretrained(
                checkpoint_folders[0], local_files_only=True
            ).cuda(FLAGS.gpu)
            model.eval()
            model.cuda_no = FLAGS.gpu

            labels_zero = get_model_accuracy(
                model, tokenizer, samples, beam_size=FLAGS.beam_size, topk=3
            )

            samples = [
                samples[i]
                for i in range(len(labels))
                if labels[i] and not labels_zero[i]
            ]

            original_result["samples"] = samples

        if len(samples) > 200:
            samples = samples[:200]
            original_result["samples"] = samples

    logging.info(f"Number of samples to evaluate is: {len(samples)}")

    original_average_scores = (
        original_result["precision"],
        original_result["recall"],
        original_result["mrr"],
    )

    logging.info(f"Original average scores: {original_average_scores}")

    baseline = rerun_baseline(samples)

    recalc_average_scores = (
        baseline["full"]["precision"],
        baseline["full"]["recall"],
        baseline["full"]["mrr"],
    )

    logging.info(f"Recalculated average scores: {recalc_average_scores}")

    random_baseline = run_random_baseline(samples)

    metrics = {"samples": samples, "evals": {}}

    metrics["evals"]["bm25plus"] = baseline
    metrics["evals"]["random"] = random_baseline

    output = FLAGS.output_metrics_prefix + ".pickle"
    with gzip.open(output, "wb") as f:
        pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    app.run(main)
