import json
import datasets
import numpy as np
from absl import app, flags, logging
from rank_bm25 import BM25Plus
from tqdm import tqdm


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "abstract_file", default=None, help="input file path to convert"
)

flags.DEFINE_string(
    "test_file", default=None, help="input file path to convert"
)

flags.DEFINE_string(
    "output_file", default=None, help="output file path to writer neighbours"
)

flags.DEFINE_integer(
    "batch_size", default=10, help="batch size to process at once"
)

flags.DEFINE_integer("topk", default=100, help="batch size to process at once")

flags.DEFINE_bool("target_only", default=False, help="targets abstracts only")

flags.DEFINE_bool(
    "only_masked_sentence",
    default=False,
    help=(
        "convert abstracts to single sentence by extracting only the masked"
        " sentence"
    ),
)


def extract_masked_sentence(abstract: str, term="<extra_id_0>"):
    term_start = abstract.find(term)
    assert term_start > -1
    term_end = term_start + len(term)
    sentence_start = abstract.rfind(". ", 0, term_start)
    if sentence_start == -1:
        sentence_start = 0
    else:
        sentence_start += 2
    sentence_end = abstract.find(". ", term_end)
    if sentence_end == -1:
        sentence_end = abstract.find(".", term_end)
    sentence_end = min(sentence_end + 1, len(abstract))
    return abstract[sentence_start:sentence_end]


def get_tokenized_query(record, extract=False):
    answer = record["targets_pretokenized"].replace("<extra_id_0> ", "")
    text = record["inputs_pretokenized"]
    if extract:
        text = extract_masked_sentence(text)
    text = text.replace("<extra_id_0>", answer).split(" ")
    return text


def get_target_equivalence_classes(abstracts):
    target_equivariance_indices = {}
    for (i, abstract) in enumerate(abstracts):
        target = (
            abstract["targets_pretokenized"]
            .replace("<extra_id_0> ", "")
            .strip()
            .lower()
        )
        if target in target_equivariance_indices:
            target_equivariance_indices[target].append(i)
        else:
            target_equivariance_indices[target] = [i]
    return target_equivariance_indices


def get_target_ids(target_ids_hashmap, record):
    target = (
        record["targets_pretokenized"]
        .replace("<extra_id_0> ", "")
        .strip()
        .lower()
    )
    return target_ids_hashmap.get(target, [0])


def main(_):
    # abstract_dataset = tf.data.TFRecordDataset(FLAGS.abstract_file)
    # abstracts = load_dataset_from_tfrecord(abstract_dataset)

    abstracts = datasets.load_dataset("data/ftrace", "abstracts", split="train")

    logging.info(f"abstracts loaded {len(abstracts)}")

    if FLAGS.target_only:
        target_ids_hashmap = get_target_equivalence_classes(abstracts)

    corpus = [
        get_tokenized_query(a, extract=FLAGS.only_masked_sentence)
        for a in abstracts
    ]
    bm25 = BM25Plus(corpus)

    test_dataset = datasets.load_dataset(
        "data/ftrace", "queries", split="train", streaming=True
    ).take(15000)

    with open(FLAGS.output_file, "w") as f:
        for example in tqdm(test_dataset):
            query = get_tokenized_query(example)
            if FLAGS.target_only:
                target_ids = np.array(
                    get_target_ids(target_ids_hashmap, example)
                )
                scores = np.array(bm25.get_batch_scores(query, target_ids))
                idxs = np.argpartition(scores, -FLAGS.topk)[-FLAGS.topk :]
                nn_idxs = idxs[np.argsort(-scores[idxs])]
                nn_scores = scores[nn_idxs].tolist()
                nn_idxs = target_ids[nn_idxs].tolist()
            else:
                scores = bm25.get_scores(query)
                idxs = np.argpartition(scores, -FLAGS.topk)[-FLAGS.topk :]
                nn_idxs = idxs[np.argsort(-scores[idxs])]
                nn_scores = scores[nn_idxs].tolist()

            neighbor_ids = abstracts.select(nn_idxs)["id"]

            line = {"scores": nn_scores, "neighbor_ids": neighbor_ids}
            print(json.dumps(line), file=f)


if __name__ == "__main__":
    app.run(main)
