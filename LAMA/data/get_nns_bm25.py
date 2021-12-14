import os
import json
import functools
import asyncio
import torch  # TODO(ekina): make this jax
import torch.nn.functional as F
from absl import app
from absl import flags
from absl import logging
from os import listdir
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi, BM25Plus

import tensorflow as tf
try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    print("Invalid device or cannot modify virtual devices once initialized.")
    raise ValueError('Cannot disable gpus for tensorflow')


FLAGS = flags.FLAGS

flags.DEFINE_string('abstract_file', default=None,
                    help='input file path to convert')

flags.DEFINE_string('test_file', default=None,
                    help='input file path to convert')

flags.DEFINE_string('output_file', default=None,
                    help='output file path to writer neighbours')

flags.DEFINE_integer('batch_size', default=10,
                     help='batch size to process at once')

flags.DEFINE_integer('topk', default=100,
                     help='batch size to process at once')

flags.DEFINE_bool('target_only', default=False,
                  help='targets abstracts only')


def get_tfexample_decoder_examples():
    """Returns tf dataset parser."""

    feature_dict = {
        'inputs_pretokenized': tf.io.FixedLenFeature([], tf.string),
        'targets_pretokenized': tf.io.FixedLenFeature([], tf.string),
        'uuid': tf.io.FixedLenFeature([], tf.string),
        'obj_uri': tf.io.FixedLenFeature([], tf.string),
        'sub_uri': tf.io.FixedLenFeature([], tf.string),
        'predicate_id': tf.io.FixedLenFeature([], tf.string),
        'obj_surface': tf.io.FixedLenFeature([], tf.string),
        'sub_surface': tf.io.FixedLenFeature([], tf.string),
       }

    def _parse_data(proto):
        data = tf.io.parse_single_example(proto, feature_dict)
        return (data['inputs_pretokenized'], data['targets_pretokenized'])

    return _parse_data


def get_tfexample_decoder_abstracts():
    """Returns tf dataset parser."""

    feature_dict = {
        'inputs_pretokenized': tf.io.FixedLenFeature([], tf.string),
        'targets_pretokenized': tf.io.FixedLenFeature([], tf.string),
        'masked_uri': tf.io.FixedLenFeature([], tf.string),
        'page_uri': tf.io.FixedLenFeature([], tf.string),
       }

    def _parse_data(proto):
        data = tf.io.parse_single_example(proto, feature_dict)
        return (data['inputs_pretokenized'], data['targets_pretokenized'])

    return _parse_data


def load_dataset_from_tfrecord(dataset):
    """Loads one shard of a dataset from the dataset file."""
    ds_loader = dataset.map(get_tfexample_decoder_abstracts()).as_numpy_iterator()
    return [d for d in ds_loader]


def get_tokenized_query(record):
    answer = record[1].decode().replace('<extra_id_0> ', '')
    q = record[0].decode().replace('<extra_id_0>', answer).split(" ")
    return q


def get_target_equivalence_classes(abstracts):
    target_equivariance_indices = {}
    for (i, abstract) in enumerate(abstracts):
        target = abstract[1].decode().replace('<extra_id_0> ', '').strip().lower()
        if target in target_equivariance_indices:
            target_equivariance_indices[target].append(i)
        else:
            target_equivariance_indices[target] = [i]
    return target_equivariance_indices


def get_target_ids(target_ids_hashmap, record):
    target = record[1].decode().replace('<extra_id_0> ', '').strip().lower()
    return target_ids_hashmap.get(target, [0])


def main(_):
    abstract_dataset = tf.data.TFRecordDataset(FLAGS.abstract_file)
    abstracts = load_dataset_from_tfrecord(abstract_dataset)
    print("abstracts loaded")
    if FLAGS.target_only:
        target_ids_hashmap = get_target_equivalence_classes(abstracts)
        
    corpus = [get_tokenized_query(a) for a in abstracts]
    bm25 = BM25Plus(corpus)

    test_dataset = tf.data.TFRecordDataset(FLAGS.test_file)
    test_loader = test_dataset.map(get_tfexample_decoder_examples()).as_numpy_iterator()

    with open(FLAGS.output_file, "w") as f:
        for example in tqdm(test_loader):
            query = get_tokenized_query(example)
            if FLAGS.target_only:
                target_ids = np.array(get_target_ids(target_ids_hashmap, example))
                scores = np.array(bm25.get_batch_scores(query, target_ids))
                idxs = np.argsort(-scores)[:FLAGS.topk]
                scores = scores[idxs].tolist()
                idxs = target_ids[idxs].tolist()           
            else:
                scores = bm25.get_scores(query)
                idxs = np.argsort(-scores)[:FLAGS.topk].tolist()
                scores = [scores[i] for i in idxs]
                
            line = {'scores': scores, 'neighbor_ids': idxs}
            print(json.dumps(line), file=f)

if __name__ == '__main__':
    app.run(main)
