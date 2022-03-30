import json
import numpy as np
from absl import app, flags
from rank_bm25 import BM25Plus
from tqdm import tqdm
from src.tf_utils import (
    get_tfexample_decoder_examples_io,
    load_dataset_from_tfrecord,
    tf,
)


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
    answer = record[1].decode().replace("<extra_id_0> ", "")
    text = record[0].decode()
    if extract:
        text = extract_masked_sentence(text)
    text = text.replace("<extra_id_0>", answer).split(" ")
    return text


def get_target_equivalence_classes(abstracts):
    target_equivariance_indices = {}
    for (i, abstract) in enumerate(abstracts):
        target = (
            abstract[1].decode().replace("<extra_id_0> ", "").strip().lower()
        )
        if target in target_equivariance_indices:
            target_equivariance_indices[target].append(i)
        else:
            target_equivariance_indices[target] = [i]
    return target_equivariance_indices


def get_target_ids(target_ids_hashmap, record):
    target = record[1].decode().replace("<extra_id_0> ", "").strip().lower()
    return target_ids_hashmap.get(target, [0])


def main(_):
    abstract_dataset = tf.data.TFRecordDataset(FLAGS.abstract_file)
    abstracts = load_dataset_from_tfrecord(abstract_dataset)

    print("abstracts loaded")

    if FLAGS.target_only:
        target_ids_hashmap = get_target_equivalence_classes(abstracts)

    corpus = [
        get_tokenized_query(a, extract=FLAGS.only_masked_sentence)
        for a in abstracts
    ]
    bm25 = BM25Plus(corpus)

    test_dataset = tf.data.TFRecordDataset(FLAGS.test_file)
    test_loader = test_dataset.map(
        get_tfexample_decoder_examples_io()
    ).as_numpy_iterator()

    with open(FLAGS.output_file, "w") as f:
        for example in tqdm(test_loader):
            query = get_tokenized_query(example)
            if FLAGS.target_only:
                target_ids = np.array(
                    get_target_ids(target_ids_hashmap, example)
                )
                scores = np.array(bm25.get_batch_scores(query, target_ids))
                idxs = np.argsort(-scores)[: FLAGS.topk]
                scores = scores[idxs].tolist()
                idxs = target_ids[idxs].tolist()
            else:
                scores = bm25.get_scores(query)
                idxs = np.argsort(-scores)[: FLAGS.topk].tolist()
                scores = [scores[i] for i in idxs]

            line = {"scores": scores, "neighbor_ids": idxs}
            print(json.dumps(line), file=f)


if __name__ == "__main__":
    app.run(main)
