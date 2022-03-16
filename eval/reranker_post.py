import gzip
import pickle
import os
from typing import Mapping, Optional, Sequence
from absl import app
from absl import flags
import numpy as np
import random
from dataclasses import dataclass, field
from absl import logging
from tqdm import tqdm
import glob

from src.metric_utils import K_EVALS, precision_recallv2, reciprocal_rankv2

FLAGS = flags.FLAGS


flags.DEFINE_string('metrics_file', default=None,
                    help='input metrics file that stores baseline '
                         'statistics and (examples, nn abstracts)')

flags.DEFINE_string('scores_folder', default=None,
                    help='output file for experiment results')

flags.DEFINE_string('output_metrics_file', default=None,
                    help="where to output metrics")

flags.DEFINE_integer('seed', default=10,
                     help="seed")

flags.DEFINE_string('exp_type', default='layers',
                    help="exp type either layers or linear combination")

EPS = 1e-7


@dataclass
class LayerConfig:
    layer_prefixes: Sequence[str]
    layer_weights: Sequence[float] = field(default_factory=list)
    index: Optional[int] = 0

    def __post_init__(self):
        if len(self.layer_weights) == 0:
            for prefix in self.layer_prefixes:
                self.layer_weights.append(1.0)


def get_all_layer_configs(num_layers=12, exp_type="layers"):
    """Returns configurations listed below"""
    if exp_type == "layers":
        layer_configs = [LayerConfig(('gradients.shared',)),
                         LayerConfig(('gradients.',))]

        layer_configs += [LayerConfig((f'gradients.encoder.block.{i}',
                                       f'gradients.decoder.block.{i}'))
                          for i in range(num_layers)]

        layer_configs += [LayerConfig(('gradients.shared',
                                       f'gradients.encoder.block.{i}',
                                       f'gradients.decoder.block.{i}'))
                          for i in range(num_layers)]

        layer_configs += [LayerConfig((f'gradients.encoder.block.{i}', ))
                          for i in range(num_layers)]

        layer_configs += [LayerConfig(('gradients.shared',
                                       f'gradients.encoder.block.{i}'))
                          for i in range(num_layers)]

        layer_configs += [LayerConfig((f'activations.encoder.block.{i}',
                                       f'activations.decoder.block.{i}'))
                          for i in range(num_layers + 1)]

        layer_configs += [LayerConfig((f'activations.encoder.block.{i}',))
                          for i in range(num_layers + 1)]

        layer_configs += [LayerConfig(('activations.encoder.block.0',
                                       'activations.decoder.block.0',
                                       f'activations.encoder.block.{i}',
                                       f'activations.decoder.block.{i}'))
                          for i in range(1, num_layers+1)]

        layer_configs += [LayerConfig(('activations.encoder.block.0',
                                       f'activations.encoder.block.{i}',
                                       f'activations.decoder.block.{i}'))
                          for i in range(1, num_layers+1)]

        layer_configs.append(LayerConfig(('activations.', )))
        layer_configs.append(LayerConfig(('activations.', 'gradients.')))
        layer_configs.append(LayerConfig(('activations.encoder.block.0',
                                          'activations.decoder.block.0',
                                          'gradients.shared')))
        layer_configs.append(LayerConfig(('activations.encoder.block.0',
                                          'gradients.shared')))
    else:
        layer_configs = []
        index = 0
        for a in np.linspace(-5, 5, num=10):
            for b in np.linspace(-5, 5, num=10):
                layer_configs.append(LayerConfig(('activations.encoder.block.0',
                                                  'activations.decoder.block.0',
                                                  'gradients.shared'),
                                                 [a, b, 1.0],
                                                 index=index))
                index += 1
        for a in (0.0, 1.0):
            for b in (0.0, 1.0):
                layer_configs.append(LayerConfig(('activations.encoder.block.0',
                                                  'activations.decoder.block.0',
                                                  'gradients.shared'),
                                                 [a, b, 1.0],
                                                 index=index))
                index += 1
    return layer_configs


def evaluate(example,
             abstracts,
             fact_abstracts,
             collapse=False):
    """Evaluate nearast abstracts to get the metrics"""
    if collapse:
        identifier = get_sentence
        _, idxs = np.unique(list(map(identifier, fact_abstracts)),
                            return_index=True)
        fact_abstracts = [fact_abstracts[id] for id in idxs]

    if len(fact_abstracts) == 0:
        logging.warning(f"empty fact abstract for query: {example}")
        return None, None, None

    # nn_ids = [a['page_uri'] for a in abstracts]
    precision, recall = precision_recallv2(abstracts,
                                           fact_abstracts,
                                           check_correct,
                                           ks=K_EVALS,
                                           collapse=collapse)
    rr = reciprocal_rankv2(abstracts,
                           fact_abstracts,
                           check_correct,
                           collapse=collapse)

    return precision, recall, rr


def rerank_with_scores(abstracts: Sequence[Mapping],
                       layer_scores: Mapping,
                       layers: Optional[LayerConfig] = None,
                       collapse: bool = False,
                       normalize: bool = False,
                       norm_type: str = "global"):
    """Given layers prefixes we sum scores of these layers
       and rerank the abstracts"""

    if layers is not None:
        # Assuming our layer configurations provide prefix codes
        def findindex(key):
            indices = np.where([key.startswith(layer)
                                for layer in layers.layer_prefixes])
            return indices[0]
        sum_pnames = [pname for pname in layer_scores[0].keys()
                      if len(findindex(pname)) > 0]

        w_weights = {pname: layers.layer_weights[findindex(pname)[0]]
                     for pname in sum_pnames}
    else:
        sum_pnames = list(layer_scores[0].keys())
        w_weights = {pname: 1.0 for pname in sum_pnames}

    abstract_scores = []
    for query_scores in layer_scores:
        if normalize:
            if norm_type == "global":
                value = 0.0
                count = 0.0
                for ptype in ("activations", "gradients"):
                    sum_pnames_typed = [pname for pname in sum_pnames
                                        if pname.startswith(ptype)]
                    if len(sum_pnames_typed) > 0:
                        num_typed_checkpoints =\
                            len(query_scores[sum_pnames_typed[0]])
                        typed_value = 0.0

                        for i in range(num_typed_checkpoints):

                            ckpt_value = [0.0, [1e-12, 1e-12]]

                            for pname in sum_pnames_typed:
                                score = query_scores[pname][i]
                                ckpt_value[0] += (score[0] * w_weights[pname])
                                ckpt_value[1][0] += score[1][0]
                                ckpt_value[1][1] += score[1][1]

                            ckpt_value =\
                                ckpt_value[0] / np.prod(np.sqrt(ckpt_value[1]))
                            typed_value += ckpt_value

                        typed_value /= num_typed_checkpoints
                        value += typed_value
                        count += 1
                if count != 0:
                    value /= count
                else:
                    value = 0
            elif norm_type == "local":
                value = 0.0
                for pname in sum_pnames:
                    pname_value = 0.0
                    for score in query_scores[pname]:
                        pname_value += score[0] / np.prod(np.sqrt(score[1]))
                    pname_value /= len(query_scores[pname])
                    value += (pname_value * w_weights[pname])
            else:
                raise ValueError(f"{norm_type} is an unknown "
                                 f"normalization type")
        else:
            value = 0.0
            for pname in sum_pnames:
                pname_value = 0.0
                for score in query_scores[pname]:
                    pname_value += score[0]
                pname_value /= len(query_scores[pname])
                value += (pname_value * w_weights[pname])

        # mean over checkpoints
        abstract_scores.append(value)

    scores = np.array(abstract_scores)
    assert len(scores) == len(abstracts), f"{len(scores)} vs {len(abstracts)}"
    # merge abstracts and scores here
    if collapse:
        scores, abstracts = collapse_abstracts_and_scores(scores, abstracts)

    sorted_idxs = np.argsort(-scores)
    abstracts_reranked = [abstracts[i] for i in sorted_idxs]
    scores_reranked = scores[sorted_idxs]

    return abstracts_reranked, scores_reranked


def collapse_abstracts_and_scores(scores: Sequence[float],
                                  abstracts: Sequence[Mapping]):
    uri_to_indices = {}
    for i, a in enumerate(abstracts):
        uri = get_sentence(a)
        if uri in uri_to_indices:
            uri_to_indices[uri].append(i)
        else:
            uri_to_indices[uri] = [i]
    uri_scores = []
    uri_indices = []
    scores = np.array(scores)
    for (uri, indices) in uri_to_indices.items():
        i_max = np.argmax(scores[indices])
        i_max = indices[i_max]
        uri_indices.append(i_max)
        uri_scores.append(scores[i_max])
    return np.array(uri_scores), [abstracts[j] for j in uri_indices]


def check_equal(a1: Mapping, a2: Mapping, collapse: bool):
    if collapse:
        return get_sentence(a1) == get_sentence(a2)
    else:
        return a1['sentence_uris'] == a2['sentence_uris']


def check_correct(a1: Mapping,
                  fact_abstracts: Sequence[Mapping],
                  collapse: bool):
    return any((check_equal(a1, a, collapse) for a in fact_abstracts))


def get_sentence(abstract):
    targets = abstract['targets_pretokenized'].replace('<extra_id_0> ', '')\
                                              .strip()
    sentence = abstract['inputs_pretokenized'].replace('<extra_id_0>', targets)
    return sentence


def average_metrics(results):
    """Average the metrics over samples"""
    metrics = {'precision': {}, 'recall': {}}
    for k in K_EVALS:
        metrics['precision'][k] = np.mean([res['precision'][k]
                                           for res in results])
        metrics['recall'][k] = np.mean([res['recall'][k]
                                        for res in results])
    metrics['mrr'] = np.mean([res['rr'] for res in results])
    metrics['samples'] = results
    return metrics


def identifier(x):
    return x['inputs_pretokenized'] + x['targets_pretokenized']


def run_all_layer_configs(samples,
                          scores,
                          num_layers=12,
                          exp_type="layers",
                          norm_type="global"):

    """Runs reranking experiments for all configurations
       listed below and returns the results"""

    assert len(scores) == len(samples), f"{len(scores)} vs {len(samples)}"

    logging.info(f"Processing {len(scores)} samples")

    layer_configs = get_all_layer_configs(num_layers, exp_type)

    results = {'cosine': {}, 'dot': {}}

    for (index, sample) in tqdm(enumerate(samples)):
        query = sample['example']

        abstracts = sample['nn_abstracts'] + sample['fact_abstracts'] + sample['distractors']

        rng = np.random.default_rng(0)
        rng.shuffle(abstracts)
        _, indices = np.unique(list(map(identifier, abstracts)),
                               return_index=True)
        abstracts = [abstracts[ind] for ind in indices]

        # Get similarity scores for all individual
        # weights x {activations, gradients, both}
        score = scores[index]

        for k, result in results.items():  # cosine or dot

            if norm_type == "global" and k == "dot":
                continue

            for method in ('collapse', 'full'):  # eval methods

                if method not in result:
                    result[method] = {}

                is_collapse = (method == 'collapse')
                is_normalized = (k == 'cosine')

                for config in layer_configs:

                    config_name = ",".join(config.layer_prefixes)

                    if exp_type != "layers":
                        config_name = config_name + "_" + str(config.index)

                    if config_name not in result[method]:
                        result[method][config_name] = []

                    abstracts_config, scores_config = rerank_with_scores(
                                                    abstracts,
                                                    score,
                                                    layers=config,
                                                    collapse=is_collapse,
                                                    normalize=is_normalized,
                                                    norm_type=norm_type)

                    precision, recall, rr = evaluate(query,
                                                     abstracts_config,
                                                     sample['fact_abstracts'],
                                                     collapse=is_collapse)

                    if precision is not None:
                        result[method][config_name].append({
                            "index": index,
                            "precision": precision,
                            "recall": recall,
                            "rr": rr,
                            "nn_abstracts": abstracts_config[:100],
                            "nn_scores": scores_config[:100].tolist(),
                            "weights": config.layer_weights,
                        })
                    else:
                        logging.warning(f"metrics are none in method: {method}")

    metrics = {'cosine': {}, 'dot': {}}
    for k, result in results.items():
        for (method_name, res) in result.items():
            metrics[k][method_name] = {}
            for (config_name, r) in res.items():
                average_result = average_metrics(r)
                metrics[k][method_name][config_name] = average_result
                logging.info(f"{config_name}\t{method_name}\t"
                             f"{k}\t{average_result['mrr']}")

    if norm_type == "global":
        metrics.pop('dot')

    return metrics


def main(_):
    for attr, flag_obj in FLAGS.__flags.items():
        logging.info("--%s=%s" % (attr, flag_obj.value))

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    with gzip.open(FLAGS.metrics_file, 'rb') as handle:
        metrics = pickle.load(handle)

    samples = metrics['samples']

    score_files = glob.glob(os.path.join(
                             FLAGS.scores_folder,
                             "*.pickle"))
    logging.info(f"score files {repr(score_files)}")
    scores = None
    for score_file in score_files:
        with gzip.open(score_file, 'rb') as handle:
            # queries[abstracts[params[dotproducts]]]
            ckpt_scores = pickle.load(handle)
            if scores is None:
                scores = [[{} for _ in abstract_scores]
                          for abstract_scores in ckpt_scores]

            for i, abstract_scores in enumerate(ckpt_scores):
                for j, score in enumerate(abstract_scores):
                    for pname, value in score.items():
                        if pname not in scores[i][j]:
                            scores[i][j][pname] = [value]
                        else:
                            scores[i][j][pname].append(value)

    global_metrics = run_all_layer_configs(samples,
                                           scores,
                                           exp_type=FLAGS.exp_type,
                                           norm_type="global")

    local_metrics = run_all_layer_configs(samples,
                                          scores,
                                          exp_type=FLAGS.exp_type,
                                          norm_type="local")

    for name, new_metrics in (("global", global_metrics),
                              ("local", local_metrics)):
        if name not in metrics['evals']:
            metrics['evals'][name] = {}
            for k, results in new_metrics.items():
                if k not in metrics['evals'][name]:
                    metrics['evals'][name][k] = {}
                for key, result in results.items():
                    metrics['evals'][name][k][key] = result

    logging.info(f"Logging to {FLAGS.output_metrics_file}")

    with gzip.open(FLAGS.output_metrics_file + ".pickle", "wb") as f:
        pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    app.run(main)
