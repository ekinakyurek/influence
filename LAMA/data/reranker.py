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
import random

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

flags.DEFINE_string('output_metrics_prefix', default=None,
                    help='output file path to write neighbours')

flags.DEFINE_string('checkpoint_folders', default=None,
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


def get_gradients_models(models, data, normalize=False, layers=None):
    grad_fn = functools.partial(get_gradients, data=data, normalize=normalize, layers=layers)
    grads = tuple(map(grad_fn, models))
    return torch.cat(grads, dim=0)


def get_gradients_models_v2(models, data, normalize=False, layers=None):
    grad_fn = functools.partial(get_gradientsv2, data=data, normalize=normalize, layers=layers)
    return tuple(map(grad_fn, models))


from torch.nn.utils import _stateless
def get_scores_batch(model, *, data, test_grad, normalize=False, layers=None):
  # TODO: get gradients by calling model.score then tf.grad sth?

    model.zero_grad()
    targets = data['labels'].cuda(model.cuda_no)
    kwargs = dict(input_ids=data['inputs'].cuda(model.cuda_no),
                  attention_mask=data['attention_mask'].cuda(model.cuda_no),
                  labels=targets)

    def model_fn(*params):
        names = list(n for n, _ in model.named_parameters())
        logits = _stateless.functional_call(model,
                                            {n: p for n, p in zip(names, params)},
                                            **kwargs).logits
        loss_batch = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100, reduction='none').view(-1, logits.size(1))
        return loss_batch.sum(dim=-1)

    scores = torch.autograd.functional.jvp(model_fn, tuple(model.parameters()), test_grad)

    return scores


def get_gradients(model, *, data, normalize=False, layers=None):
  # TODO: get gradients by calling model.score then tf.grad sth?
    model.zero_grad()

    model(input_ids=data['inputs'].cuda(model.cuda_no),
          labels=data['targets'].cuda(model.cuda_no)).loss.backward()

    if layers is not None:
        grad = []
        for layer in layers:
            keys = layer.split('/')
            block = model._modules[keys[0]]
            for key in keys[1:]:
                block = block._modules[key]
            grad.extend((p.grad.detach().flatten().cpu() for p in block.parameters()))
    else:
        grad = (param.grad.detach().flatten().cpu() for param in model.parameters())

    if normalize:
        grad = list(map(f_normalize, grad))

    if type(grad) != list:
        grad = list(grad)

    return torch.cat(grad, dim=0)


def get_gradientsv2(model, *, data, normalize=False, layers=None):
  # TODO: get gradients by calling model.score then tf.grad sth?
    model.zero_grad()

    model(input_ids=data['inputs'].cuda(model.cuda_no),
          labels=data['targets'].cuda(model.cuda_no)).loss.backward()
    #
    # if layers is not None:
    #     grad = []
    #     for layer in layers:
    #         keys = layer.split('/')
    #         block = model._modules[keys[0]]
    #         for key in keys[1:]:
    #             block = block._modules[key]
    #         grad.extend((p.grad.detach().flatten().cpu() for p in block.parameters()))
    # else:
    grad = (param.grad.detach() for param in model.parameters())

    # if normalize:
    #     grad = list(map(f_normalize, grad))

    if type(grad) != tuple:
        grad = tuple(grad)

    return grad


def sharded_open(filename):
    if '@' in filename:
        prefix, no_shards = filename.split('@')
        no_shards = int(no_shards)
        filenames = [f'{prefix}-{str(i).zfill(5)}-of-{str(no_shards).zfill(5)}' for i in range(no_shards)]
        dataset = tf.data.TFRecordDataset(filenames=filenames)
    else:
        dataset = tf.data.TFRecordDataset(filename)
    return dataset


from torch.utils.data import Dataset, DataLoader

class AbstractsDataset(Dataset):
    def __init__(self, abstracts, tokenizer, max_input_length=768, max_target_length=224):
        self.abstracts = abstracts
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.abstracts)

    def __getitem__(self, idx):
        return self.abstracts[idx]

    def collate_fn(self, data):
        input_sequences = [datum['inputs_pretokenized'] for datum in data]
        target_sequences = [datum['targets_pretokenized'] for datum in data]
        encoding = self.tokenizer(input_sequences,
                                  padding='longest',
                                  max_length=self.max_input_length,
                                  truncation=True,
                                  return_tensors="pt")

        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        target_encoding = self.tokenizer(target_sequences,
                                         padding='longest',
                                         max_length=self.max_target_length,
                                         truncation=True)

        labels = target_encoding.input_ids

        labels = [
               [(label if label != self.tokenizer.pad_token_id else -100)
                    for label in labels_example]
                        for labels_example in labels]

        labels = torch.tensor(labels)

        return {'inputs': input_ids, 'labels': labels, 'attention_mask': attention_mask}




def rerank(models, tokenizer, query, abstracts, batch_size=5, **kwargs):
    query_grad = get_gradients_models(models,
                                      data=get_tokenized_query(tokenizer,
                                                               query),
                                      **kwargs)

    scores = [torch.dot(query_grad,
                        get_gradients_models(models,
                                             data=get_tokenized_query(tokenizer, a),
                                             **kwargs))
              for a in abstracts]

    # dataset = AbstractsDataset(abstracts, tokenizer)
    # dataloader = DataLoader(dataset,
    #                         batch_size=batch_size,
    #                         collate_fn=dataset.collate_fn)
    #
    # scores = []
    # for data in dataloader:
    #     scores_batch = 0
    #     for i, model in enumerate(models):
    #         model_score = get_scores_batch(model,
    #                                        data=data,
    #                                        test_grad=query_grad[i],
    #                                        **kwargs)[1]
    #         scores_batch += model_score.cpu().detach().numpy()
    #         pdb.set_trace()
    #     scores.extend(scores_batch)

    scores = np.array(scores)
    sorted_idxs = np.argsort(-scores)
    abstracts_reranked = [abstracts[i] for i in sorted_idxs]
    scores_reranked = scores[sorted_idxs]
    return abstracts_reranked, scores_reranked


def precision_recall(nearest_ids, correct_ids, ks=(1, 5, 10, 50, 100)):
    precision, recall = {}, {}
    for k in ks:
        nn_k = nearest_ids[:k]
        precision[k] = len([1 for id in nn_k if id in correct_ids]) / k
        recall[k] = len([1 for id in correct_ids if id in nn_k]) / len(correct_ids)
    return precision, recall


def reciprocal_rank(nearest_ids, correct_ids):
    for i, id in enumerate(nearest_ids):
        if id in correct_ids:
            return 1 / (i+1)
    return 0


def evaluate(example, abstracts, hashmap):
    key = ",".join((example['predicate_id'],
                    example['obj_uri'],
                    example['sub_uri']))
    uris = hashmap.get(key, None)
    if uris is None or len(uris) == 0:
        return None, None, None
    #
    nn_ids = [a['page_uri'] for a in abstracts]
    precision, recall = precision_recall(nn_ids, uris, ks=(1, 5, 10, 50, 100))
    rr = reciprocal_rank(nn_ids, uris)
    return precision, recall, rr

def dict_diff(d1, d2):
    return {k: v-d2[str(k)] for (k, v) in d1.items()}

def run_a_layer_config(models, tokenizer, hashmap, samples, layers=None, normalize=True):
    # precisions = []
    # recalls = []
    # rrs = []
    results = []
    for (i, sample) in tqdm(enumerate(samples)):
        query = sample['example']

        abstracts = sample['nn_abstracts'] + sample['fact_abstracts'] + sample['distractors']

        # abstracts = np.unique(abstracts)

        #
        # query_grad = get_gradients_models(models,
        #                 data=get_tokenized_query(tokenizer, query),
        #                 normalize=normalize)

        #

        abstracts, scores = rerank(models, tokenizer, query, abstracts, normalize=normalize, layers=layers)

        precision, recall, rr = evaluate(query, abstracts[:100], hashmap)
        # if precision is not None and recall is not None:
        #     precisions.append(precision)
        #     recalls.append(recall)
        # rrs.append(rr)
        results.append({
            "example": sample["example"],
            "precision": precision,
            "recall": recall,
            "rr": rr,
            "nn_abstracts": abstracts[:100],
            "nn_scores": scores[:100].tolist(),
        })
        # nn_abstracts.appen(abstracts[:100])
        # nn_scores.append(scores[:100])
        #
        #

    metrics = {'precision': {}, 'recall': {}}
    for k in (1, 5, 10,  50, 100):
        metrics['precision'][k] = np.mean([res['precision'][k] for res in results])
        metrics['recall'][k] = np.mean([res['recall'][k] for res in results])
    metrics['mrr'] = np.mean([res['rr'] for res in results])
    metrics['samples'] = results
    return metrics


def main(_):
    # if FLAGS.layers is not None:
    #     FLAGS.layers = FLAGS.layers.split(',')
    np.random.seed(10)
    random.seed(10)
    metrics_results = json.load(open(FLAGS.metrics_file))
    hashmap = json.load(open(FLAGS.hashmap_file, 'r'))

    checkpoint_folders = FLAGS.checkpoint_folders.split(',')
    models = []
    for i, folder in enumerate(checkpoint_folders):
        models.append(MT5ForConditionalGeneration.from_pretrained(
                                                folder,
                                                local_files_only=True).cuda(i)
                      )
        models[-1].eval()
        models[-1].cuda_no = i

    tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")

    samples = metrics_results['samples']
    results = ['layer_no\tnormalize\tprecision\trecall\tmrr']
    print(results[0])
    #for layer_no in range(0, 1):
    layer_no = 'special'
    for normalize in (True, False):
        if layer_no == -1:
            layers = ['shared']
        elif layer_no == 12:
            layers = None
        elif layer_no == 'special':
            layers = ['shared', 'encoder/block/0', 'decoder/block/0']
        else:
            layers = f'encoder/block/{layer_no},decoder/block/{layer_no}'.split(',')

        metrics = run_a_layer_config(models,
                                     tokenizer,
                                     hashmap,
                                     samples,
                                     layers=layers,
                                     normalize=normalize)
        #
        line = f'{layer_no}\t{normalize}\t{metrics["precision"]}\t{metrics["recall"]}\t{metrics["mrr"]}'
        print(line)
        with open(FLAGS.output_metrics_prefix + f"_normalized_{normalize}_layer_{layer_no}.json", "w") as f:
            json.dump(metrics, f)



if __name__ == '__main__':
    app.run(main)
