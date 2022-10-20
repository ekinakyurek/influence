from typing import Callable, List, Mapping
import tensorflow as tf


try:
    # Disable all GPUS
    tf.config.set_visible_devices([], "GPU")
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != "GPU"
except AssertionError:
    print("Invalid device or cannot modify virtual devices once initialized.")
    raise ValueError("Cannot disable gpus for tensorflow")


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_tfexample_decoder_examples_io() -> Callable:
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


def get_tfexample_decoder() -> Callable:
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
        return tf.io.parse_single_example(proto, feature_dict)

    return _parse_data


def _tfrecord(record):
    feature = {
        k: _bytes_feature(v.encode("utf-8")) for (k, v) in record.items()
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def get_tfexample_decoder_abstracts() -> Callable:
    """Returns tf dataset parser."""

    feature_dict = {
        "inputs_pretokenized": tf.io.FixedLenFeature([], tf.string),
        "targets_pretokenized": tf.io.FixedLenFeature([], tf.string),
        "masked_uri": tf.io.FixedLenFeature([], tf.string),
        "page_uri": tf.io.FixedLenFeature([], tf.string),
        "masked_type": tf.io.FixedLenFeature([], tf.string),
        "facts": tf.io.FixedLenFeature([], tf.string),
        "sentence_uris": tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_data(proto):
        data = tf.io.parse_single_example(proto, feature_dict)
        return (data["inputs_pretokenized"], data["targets_pretokenized"])

    return _parse_data


def get_tfexample_decoder_abstracts_v2() -> Callable:
    """Return tf dataset parser for abstracts."""

    feature_dict = {
        "inputs_pretokenized": tf.io.FixedLenFeature([], tf.string),
        "targets_pretokenized": tf.io.FixedLenFeature([], tf.string),
        "masked_uri": tf.io.FixedLenFeature([], tf.string),
        "page_uri": tf.io.FixedLenFeature([], tf.string),
        "masked_type": tf.io.FixedLenFeature([], tf.string),
        "facts": tf.io.FixedLenFeature([], tf.string),
        "sentence_uris": tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_data(proto):
        data = tf.io.parse_single_example(proto, feature_dict)
        return data

    return _parse_data


def load_examples_from_tf_records(file: str) -> List[Mapping]:
    """Loads one shard of a dataset from the dataset file."""
    dataset = tf.data.TFRecordDataset(file)
    ds = dataset.map(get_tfexample_decoder()).as_numpy_iterator()
    return [{k: v.decode() for k, v in datum.items()} for datum in ds]


def load_dataset_from_tfrecord(dataset) -> List:
    """Loads one shard of a dataset from the dataset file."""
    ds_loader = dataset.map(
        get_tfexample_decoder_abstracts()
    ).as_numpy_iterator()
    return [d for d in ds_loader]


def create_dummy_tfrecord_dataset(output_file, n, feature_size):
    """Creates dummy data to test this code."""
    # Somewhat this is extremely slow!
    with tf.io.TFRecordWriter(output_file) as writer:
        for i in range(n):
            value = tf.random.uniform((feature_size,))
            tf_example = _tfrecord_old(i, value)
            writer.write(tf_example.SerializeToString())


def dump_abstracts_tf_record(abstracts, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for abstract in abstracts:
            tf_example = _tfrecord(abstract)
            writer.write(tf_example.SerializeToString())


# For previous nn code
def get_index_decoder(feature_length):
    """Returns tf dataset parser."""
    vector_feature_description = {
        "index": tf.io.FixedLenFeature([1], tf.int64),
        "embedding": tf.io.FixedLenFeature([feature_length], tf.float32),
    }

    def _parse_data(proto):
        data = tf.io.parse_single_example(proto, vector_feature_description)
        return data["index"]

    return _parse_data


def get_tfexample_decoder_old(feature_length):
    """Returns tf dataset parser."""
    print("feature_length: ", feature_length)
    vector_feature_description = {
        "index": tf.io.FixedLenFeature([1], tf.int64),
        "embedding": tf.io.FixedLenFeature([feature_length], tf.float32),
    }

    def _parse_data(proto):
        data = tf.io.parse_single_example(proto, vector_feature_description)
        return data["embedding"]

    return _parse_data


def _tfrecord_old(index, value):
    """Converts dummy data values to tf records."""
    record = {
        "embedding": tf.train.Feature(
            float_list=tf.train.FloatList(value=value)
        )
    }
    return tf.train.Example(features=tf.train.Features(feature=record))


def sharded_open(filename: str):
    """Open sharded tfrecords as TFRecordDataset"""
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
