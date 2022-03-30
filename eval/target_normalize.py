import asyncio
import contextlib2
import numpy as np
import torch  # TODO(ekina): make this jax
from absl import app, flags
from src.linalg_utils import normalize_segments
from src.tf_utils import get_index_decoder, get_tfexample_decoder_old, tf


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "abstract_vectors", default=None, help="input file path to convert"
)

flags.DEFINE_string(
    "test_vectors", default=None, help="input file path to convert"
)

flags.DEFINE_string(
    "abstract_file", default=None, help="input file path to convert"
)

flags.DEFINE_string(
    "test_file", default=None, help="input file path to convert"
)

flags.DEFINE_string(
    "output_file", default=None, help="output file path to writer neighbours"
)

flags.DEFINE_string(
    "test_output_file",
    default=None,
    help="output file path to writer neighbours",
)

flags.DEFINE_integer("feature_size", default=4 * 2048, help="size of vectors")

flags.DEFINE_bool("normalize", default=False, help="normalize embeddings")

flags.DEFINE_integer(
    "gpu_workers", default=1, help="number of gpus to distribute"
)


def load_a_shard_from_tfrecord(
    dataset, feature_length, positions, normalize=False
):
    """Loads one shard of a dataset from the dataset file."""
    shard_length = positions[1] - positions[0]
    ds_loader = (
        dataset.skip(positions[0])
        .take(shard_length)
        .map(get_tfexample_decoder_old(feature_length))
        .batch(shard_length, num_parallel_calls=4)
        .as_numpy_iterator()
    )

    X = next(iter(ds_loader))
    X = torch.from_numpy(X)
    if normalize:
        X = normalize_segments(X, size=1024)
    return X


def load_a_shard_from_numpy(
    np_file, feature_length, positions, normalize=False
):
    """Loads one shard of a dataset from the dataset file."""
    X = np.load(np_file, mmap_mode="r")
    X = torch.from_numpy(X[positions[0] : positions[1], :])
    if normalize:
        X = normalize_segments(X, size=1024)
    return X


def get_shard_positions(dataset, n_workers):
    """Gets start and end indicies of a shard in the merged dataset."""
    if type(dataset) == str:
        total_abstracts = FLAGS.fake_data_size
    else:
        total_abstracts = sum(1 for _ in dataset)
    shard_length = total_abstracts // n_workers
    positions = []
    start = 0
    for i in range(n_workers):
        current_start = start
        if i == n_workers - 1:
            current_end = total_abstracts
        else:
            current_end = current_start + shard_length
        positions.append((current_start, current_end))
        start = current_end
    return positions


def sharded_open(filename):
    if "@" in filename:
        prefix, no_shards = filename.split("@")
        no_shards = int(no_shards)
        filenames = [
            f"{prefix}-{str(i).zfill(5)}-of-{str(no_shards).zfill(5)}"
            for i in range(no_shards)
        ]
        dataset = tf.data.TFRecordDataset(filenames=filenames)
    else:
        dataset = tf.data.TFRecordDataset(filename)
    return dataset


async def _prepare_shards(file, args):
    """Reads shards from the merged file and sends them to GPUs."""

    #  dataset = args.abstract_vectors
    # Create a dictionary describing the features.
    dataset = sharded_open(file)

    async def get_a_shard(i, index):
        shard = load_a_shard_from_tfrecord(
            dataset, args.feature_size, index, normalize=FLAGS.normalize
        )

        return shard

    positions = get_shard_positions(dataset, args.gpu_workers)

    loaders = [get_a_shard(*s) for s in enumerate(positions)]

    shards = await asyncio.gather(*loaders)

    return shards, positions


def get_indices(dataset, feature_length):
    ds = dataset.map(get_index_decoder(feature_length)).as_numpy_iterator()
    return np.array([d[0] for d in ds])


def get_tfexample_decoder_old_examples():
    """Returns tf dataset parser."""

    feature_dict = {
        "inputs_pretokenized": tf.io.FixedLenFeature([], tf.string),
        "targets_pretokenized": tf.io.FixedLenFeature([], tf.string),
        "uuid": tf.io.FixedLenFeature([], tf.string),
        "obj_uri": tf.io.FixedLenFeature([], tf.string),
        "sub_uri": tf.io.FixedLenFeature([], tf.string),
        "predicate_id": tf.io.FixedLenFeature([], tf.string),
        "obj_surface": tf.io.FixedLenFeature([], tf.string),
        "sub_surface": tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_data(proto):
        data = tf.io.parse_single_example(proto, feature_dict)
        return (data["inputs_pretokenized"], data["targets_pretokenized"])

    return _parse_data


def get_tfexample_decoder_old_abstracts():
    """Returns tf dataset parser."""

    feature_dict = {
        "inputs_pretokenized": tf.io.FixedLenFeature([], tf.string),
        "targets_pretokenized": tf.io.FixedLenFeature([], tf.string),
        "masked_uri": tf.io.FixedLenFeature([], tf.string),
        "page_uri": tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_data(proto):
        data = tf.io.parse_single_example(proto, feature_dict)
        return (data["inputs_pretokenized"], data["targets_pretokenized"])

    return _parse_data


def load_abstracts_from_tfrecord(dataset):
    """Loads one shard of a dataset from the dataset file."""
    ds_loader = dataset.map(
        get_tfexample_decoder_old_abstracts()
    ).as_numpy_iterator()
    return [d for d in ds_loader]


def load_examples_from_tfrecord(dataset):
    """Loads one shard of a dataset from the dataset file."""
    ds_loader = dataset.map(
        get_tfexample_decoder_old_examples()
    ).as_numpy_iterator()
    return [d for d in ds_loader]


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _tfrecord(index, value):
    """Converts dummy data values to tf records."""
    record = {
        "index": _int64_feature(index),
        "embedding": tf.train.Feature(
            float_list=tf.train.FloatList(value=value)
        ),
    }
    return tf.train.Example(features=tf.train.Features(feature=record))


def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    """Opens all TFRecord shards for writing and adds them to an exit stack.
    Args:
      exit_stack: A context2.ExitStack used to automatically closed the TFRecords
        opened in this function.
      base_path: The base path for all shards
      num_shards: The number of shards
    Returns:
      The list of opened TFRecords. Position k in the list corresponds to shard k.
    """
    tf_record_output_filenames = [
        "{}-{:05d}-of-{:05d}".format(base_path, idx, num_shards)
        for idx in range(num_shards)
    ]

    tfrecords = [
        exit_stack.enter_context(tf.io.TFRecordWriter(file_name))
        for file_name in tf_record_output_filenames
    ]

    return tfrecords


def class_projection(vectors, class_vector):
    # class_norm = np.linalg.norm(class_vector)
    # return class_vector[None, :] * np.dot(vectors, class_vector[:,  None]) / class_norm
    return class_vector


def get_sorted_records(vector_file, data_file, load_fn, feature_size):
    abstract_indices = get_indices(sharded_open(vector_file), feature_size)
    abstract_dataset = tf.data.TFRecordDataset(data_file)
    abstracts = np.array(load_fn(abstract_dataset))
    abstracts = abstracts[abstract_indices]
    return abstracts, abstract_indices


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


def write_sharded_files(output_file, vectors, indices):
    out_prefix, num_shards = output_file.split("@")
    num_shards = int(num_shards)
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = open_sharded_output_tfrecords(
            tf_record_close_stack, out_prefix, num_shards
        )
        for i, (index, vector) in enumerate(zip(indices, vectors)):
            example = _tfrecord(index, vector).SerializeToString()
            output_shard_index = i % num_shards
            output_tfrecords[output_shard_index].write(example)


def main(_):
    abstracts, abstract_indices = get_sorted_records(
        FLAGS.abstract_vectors,
        FLAGS.abstract_file,
        load_abstracts_from_tfrecord,
        FLAGS.feature_size,
    )

    test_examples, test_indices = get_sorted_records(
        FLAGS.test_vectors,
        FLAGS.test_file,
        load_examples_from_tfrecord,
        FLAGS.feature_size,
    )

    target_equivariance_indices = get_target_equivalence_classes(abstracts)
    test_target_equivariance_indices = get_target_equivalence_classes(
        test_examples
    )

    shards, _ = asyncio.run(_prepare_shards(FLAGS.abstract_vectors, FLAGS))
    shards_test, _ = asyncio.run(_prepare_shards(FLAGS.abstract_vectors, FLAGS))

    assert len(shards) == 1 and len(shards_test) == 1

    vectors = shards[0]
    test_vectors = shards_test[0]

    for target, idxs in target_equivariance_indices.items():
        class_vector = torch.mean(vectors[idxs, :], dim=0)
        # class_norm = np.linalg.norm(class_vector)
        vectors[idxs, :] -= class_projection(vectors[idxs, :], class_vector)

        if target in test_target_equivariance_indices:
            test_idxs = test_target_equivariance_indices[target]
            test_vectors[test_idxs, :] -= class_projection(
                test_vectors[test_idxs, :], class_vector
            )

    write_sharded_files(FLAGS.output_file, vectors, abstract_indices)
    write_sharded_files(FLAGS.test_output_file, test_vectors, test_indices)


if __name__ == "__main__":
    app.run(main)
