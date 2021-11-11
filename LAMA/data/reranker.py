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
from transformers import MT5ForConditionalGeneration, T5Tokenizer
import pdb


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

flags.DEFINE_string('metrics_file', default=None,
                    help='output file path to writer neighbours')

flags.DEFINE_string('checkpoint_prefix', default=None,
                    help='checkpoint_prefix')

flags.DEFINE_string('layers', default=None,  # 'encoder/block/0'
                    help='layers used in reranking')

flags.DEFINE_string('hashmap_file', default=None,
                    help='hashmap that maps relation,obj,subj->page_uris')

flags.DEFINE_bool('normalize', default=False,
                  help="normalize embeddings")

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
        return data  # (data['inputs_pretokenized'], data['targets_pretokenized'])

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
        return data  # (data['inputs_pretokenized'], data['targets_pretokenized'])

    return _parse_data

def load_dataset_from_tfrecord(dataset, load_fn):
    """Loads one shard of a dataset from the dataset file."""
    ds_loader = dataset.map(load_fn()).as_numpy_iterator()
    return [d for d in ds_loader]


def get_tokenized_query(tokenizer, record):
    q = {'inputs': tokenizer(record['inputs_pretokenized'],
                            return_tensors='pt').input_ids,
         'targets': tokenizer(record['targets_pretokenized'],
                             return_tensors='pt').input_ids}
    return q



def f_normalize(x):
    return F.normalize(x, dim=0)


def get_gradients(model, data, normalize=False, layers=None):
  # TODO: get gradients by calling model.score then tf.grad sth?
    model.zero_grad()

    model(input_ids=data['inputs'].cuda(),
          labels=data['targets'].cuda()).loss.backward()

    if layers is not None:
        grad = []
        for layer in layers:
            keys = layer.split('/')
            block = model._modules[keys[0]]
            for key in keys[1:]:
                block = block._modules[key]
            grad.extend([p.grad.detach().flatten().cpu() for p in block.parameters()])
    else:
        grad = [param.grad.detach().flatten().cpu() for param in model.parameters()]

    if normalize:
        grad = list(map(f_normalize, grad))

    return torch.cat(grad, dim=0)


def sharded_open(filename):
    if '@' in filename:
        prefix, no_shards = filename.split('@')
        no_shards = int(no_shards)
        filenames = [f'{prefix}-{str(i).zfill(5)}-of-{str(no_shards).zfill(5)}' for i in range(no_shards)]
        dataset = tf.data.TFRecordDataset(filenames=filenames)
    else:
        dataset = tf.data.TFRecordDataset(filename)
    return dataset


def rerank(model, tokenizer, query, abstracts, **kwargs):
    query_grad = get_gradients(model,
                               get_tokenized_query(tokenizer, query), **kwargs)
    scores = [torch.dot(query_grad,
                        get_gradients(model,
                                      get_tokenized_query(tokenizer, a), **kwargs))
              for a in abstracts]

    scores = np.array(scores)
    sorted_idxs = np.argsort(-scores)
    abstracts_reranked = [abstracts[i] for i in sorted_idxs]
    scores_reranked = scores[sorted_idxs]
    return abstracts_reranked, scores_reranked


def evaluate(example, abstracts, hashmap):
    key = ",".join((example['predicate_id'],
                    example['obj_uri'],
                    example['sub_uri']))
    uris = hashmap.get(key, None)
    if uris is None:
        return None, None
    precision = {}
    recall = {}
    # pdb.set_trace()
    nn_ids = [a['page_uri'] for a in abstracts]
    for k in (1, 5, 10, 50, 100):
        nn_k = nn_ids[:k]
        precision[k] = len([1 for id in nn_k if id in uris]) / len(nn_k)
        recall[k] = len([1 for id in uris if id in nn_k]) / len(uris)

    return precision, recall

def dict_diff(d1, d2):
    return {k: v-d2[str(k)] for (k, v) in d1.items()}

def run_a_layer_config(model, tokenizer, hashmap, samples, layers=None, normalize=True):
    precisions = []
    recalls = []
    for (i, sample) in tqdm(enumerate(samples)):
        query = sample['example']

        abstracts = sample['nn_abstracts']  # + sample['fact_abstracts']

        query_grad = get_gradients(model,
                        get_tokenized_query(tokenizer, query),
                        normalize=normalize)

        # pdb.set_trace()

        abstracts, scores = rerank(model, tokenizer, query, abstracts, normalize=normalize, layers=layers)

        precision, recall = evaluate(query, abstracts[:100], hashmap)
        if precision is not None and recall is not None:
            precisions.append(dict_diff(precision, sample['precision']))
            recalls.append(dict_diff(recall, sample['recall']))
        # pdb.set_trace()
        # pdb.set_trace()

    metrics = {'precision': {}, 'recall': {}}
    for k in (1, 5, 10,  50, 100):
        metrics['precision'][k] = np.mean([p[k] for p in precisions])
        metrics['recall'][k] = np.mean([r[k] for r in recalls])
    return metrics


def main(_):
    # if FLAGS.layers is not None:
    #     FLAGS.layers = FLAGS.layers.split(',')

    metrics_results = json.load(open(FLAGS.metrics_file))
    hashmap = json.load(open(FLAGS.hashmap_file, 'r'))

    model = MT5ForConditionalGeneration.from_pretrained(
                                                FLAGS.checkpoint_prefix,
                                                local_files_only=True).cuda()
    model.eval()

    tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")

    samples = metrics_results['samples']
    results = ['layer_no\tnormalize\tprecision\trecall']
    print(results[0])
    for layer_no in range(-1, 2):
        for normalize in (True, False):
            if layer_no == -1:
                layers = ['shared']
            elif layer_no == 12:
                layers = None
            else:
                layers = f'encoder/block/{layer_no},decoder/block/{layer_no}'.split(',')

            metrics = run_a_layer_config(model,
                                         tokenizer,
                                         hashmap,
                                         samples,
                                         layers=layers,
                                         normalize=normalize)
            # pdb.set_trace()
            line = f'{layer_no}\t{normalize}\t{metrics["precision"]}\t{metrics["recall"]}'
            print(line)
            results.append(line)

    # print("\n".join(results))

    # pdb.set_trace()




if __name__ == '__main__':
    app.run(main)
