import gzip
import json
import pickle
import random
import datasets
import numpy as np
from absl import app, flags, logging
from tqdm import tqdm
from src.metric_utils import K_EVALS, precision_recall, reciprocal_rank


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "nn_list_file",
    default=None,
    help="nearest neighbors of the test examples in the same order",
)

flags.DEFINE_string(
    "fact_to_ids_file",
    default=None,
    help="fact_to_ids that maps relation,obj,subj->page_uris",
)

flags.DEFINE_string(
    "output_file", default=None, help="output file path to writer neighbours"
)

flags.DEFINE_bool("target_only", default=False, help="targets abstracts only")

flags.DEFINE_boolean("disable_tqdm", False, help="Disable tqdm")

flags.DEFINE_integer("seed", 10, help="Random seed")


# def extract_masked_sentence(abstract: str, term="<extra_id_0>"):
#     term_start = abstract.find(term)
#     assert term_start > -1
#     term_end = term_start + len(term)
#     sentence_start = abstract.rfind(". ", 0, term_start)
#     if sentence_start == -1:
#         sentence_start = 0
#     else:
#         sentence_start += 2
#     sentence_end = abstract.find(". ", term_end)
#     if sentence_end == -1:
#         sentence_end = abstract.find(".", term_end)
#     sentence_end = min(sentence_end + 1, len(abstract))
#     return abstract[sentence_start:sentence_end]


def main(_):
    examples = datasets.load_dataset(
        "data/ftrace", "queries", split="train"
    ).select(range(15000))

    abstracts = datasets.load_dataset("data/ftrace", "abstracts", split="train")
    ids_to_abstracts_keys = abstracts["id"]

    logging.info(f"ids to abstracts length {len(ids_to_abstracts_keys)}")

    logging.info("Building ids to abstracts hashmap")
    ids_to_abstracts = {
        key: abstracts[i] for i, key in enumerate(ids_to_abstracts_keys)
    }

    logging.info(f"Length of abstracts {len(ids_to_abstracts)}")

    with open(FLAGS.fact_to_ids_file, "r") as handle:
        fact_to_ids = json.load(handle)

    nns = []
    with open(FLAGS.nn_list_file, "r") as f:
        for line in f:
            nns.append(json.loads(line))

    assert len(nns) == len(examples)

    metrics = {"precision": {}, "recall": {}, "samples": []}

    precisions, recalls, rrs = [], [], []

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    indices = np.random.permutation(len(examples)).tolist()
    counted = 0
    logging.info("Building targets to abstracts idxs hashmap")
    targets_to_idxs = {}
    for idx, a in enumerate(abstracts):
        target = a["targets_pretokenized"]
        if target in targets_to_idxs:
            targets_to_idxs[target].append(idx)
        else:
            targets_to_idxs[target] = [idx]

    for index in tqdm(indices, disable=FLAGS.disable_tqdm):
        nn = nns[index]
        example = examples[index]
        nn_ids = nn["neighbor_ids"]

        fact = ",".join(
            (example["predicate_id"], example["obj_uri"], example["sub_uri"])
        )

        fact_ids = list(map(str, fact_to_ids.get(fact, [])))

        if len(fact_ids) == 0:
            continue

        counted += 1

        precision, recall = precision_recall(nn_ids, fact_ids, ks=K_EVALS)
        rr = reciprocal_rank(nn_ids, fact_ids)
        precisions.append(precision)
        recalls.append(recall)
        rrs.append(rr)

        # pdb.set_trace()

        if len(metrics["samples"]) < 10000:

            facts_abstracts = [ids_to_abstracts[str(id)] for id in fact_ids]

            if len(facts_abstracts) == 0:
                continue

            targets = example["targets_pretokenized"]
            if FLAGS.target_only:
                target_idxs = targets_to_idxs.get(targets, [])
                distractor_idxs = np.random.choice(
                    target_idxs,
                    min(200, len(target_idxs)),
                    replace=False,
                ).tolist()
                distractors = list(abstracts.select(distractor_idxs))
            else:
                target_idxs = targets_to_idxs.get(targets, [])
                distractor_idxs = np.random.choice(
                    target_idxs,
                    min(100, len(target_idxs)),
                    replace=False,
                ).tolist()
                distractors = list(abstracts.select(distractor_idxs))

                ext_ids = np.random.choice(
                    ids_to_abstracts_keys, 100, replace=False
                )
                distractors += [ids_to_abstracts[id] for id in ext_ids]

            nn_abstracts = [ids_to_abstracts[id] for id in nn_ids]

            metrics["samples"].append(
                {
                    "example": example,
                    "precision": precision,
                    "recall": recall,
                    "rr": rrs[-1],
                    "nn": nn,
                    "nn_abstracts": nn_abstracts,
                    "fact_abstracts": facts_abstracts,
                    "distractors": distractors,
                }
            )
        else:
            break

    for k in K_EVALS:
        metrics["precision"][k] = np.mean([p[k] for p in precisions])
        metrics["recall"][k] = np.mean([r[k] for r in recalls])

    metrics["mrr"] = np.mean(rrs)

    logging.info(f"MRR: {metrics['mrr']}")

    with gzip.open(FLAGS.output_file, "w") as f:
        pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    app.run(main)
