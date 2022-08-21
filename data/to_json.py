import json
from typing import Callable
from absl import app, flags
from src.tf_utils import tf


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_file", default=None, help="input file path to convert"
)
flags.mark_flag_as_required("input_file")


def get_tfexample_decoder_examples_io() -> Callable:
    """Returns tf dataset parser."""

    feature_dict = {
        "inputs": tf.io.FixedLenFeature([], tf.string),
        "targets": tf.io.FixedLenFeature([], tf.string),
        "guid": tf.io.FixedLenFeature([], tf.string),
        "proponent_guids": tf.io.VarLenFeature(tf.string),
    }

    def _parse_data(proto):
        data = tf.io.parse_single_example(proto, feature_dict)
        return (
            data["inputs"],
            data["targets"],
            data["guid"],
            data["proponent_guids"],
        )

    return _parse_data


def load_dataset_from_tfrecord(
    filename,
):
    """Loads one shard of a dataset from the dataset file."""

    dataset = tf.data.TFRecordDataset(
        filename,
    )
    ds_loader = dataset.map(get_tfexample_decoder_examples_io())

    def to_dict(data):
        processed_data = {
            "inputs_pretokenized": data[0].numpy().decode(),
            "targets_pretokenized": data[1].numpy().decode(),
            "uuid": data[2].numpy().decode(),
            "proponents": tf.sparse.to_dense(data[3])
            .numpy()
            .astype(str)
            .tolist(),
        }
        return processed_data

    return [to_dict(data) for data in ds_loader]


def main(argv):
    data = load_dataset_from_tfrecord(FLAGS.input_file)
    # write to json
    js_file = FLAGS.input_file.replace(".tfrecord", ".jsonl")
    with open(js_file, "w") as f:
        for example in data:
            json.dump(example, f)
            f.write("\n")


if __name__ == "__main__":
    app.run(main)
