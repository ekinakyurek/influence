from dataclasses import dataclass
import json
from typing import Optional, Sequence, Mapping
from numpy.lib.function_base import average
import torch  # TODO(ekina): make this jax
import torch.nn.functional as F
from absl import app
from absl import flags
from absl import logging
import numpy as np
from tqdm import tqdm
from transformers import MT5ForConditionalGeneration, T5Tokenizer
import random
import tensorflow as tf
import copy
from dataclasses import dataclass, field
# import pdb

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
                    help='input metrics file that stores baseline statistics and (examples, nn abstracts)')

flags.DEFINE_string('output_metrics_prefix', default=None,
                    help='output file for experiment results')

flags.DEFINE_string('checkpoint_folders', default=None,
                    help='list of checkpoint folders (coma spereated)')

flags.DEFINE_string('layers', default=None,  # 'encoder/block/0'
                    help='layers used in reranking (deprecated)')

flags.DEFINE_string('hashmap_file', default=None,
                    help='hashmap that maps relation,obj,subj->sentence_uris')

flags.DEFINE_string('exp_type', default='layers', 
                    help="exp type either layers or linear combination")

flags.DEFINE_integer('beam_size', default=3,
                     help="beam size for accuracy calculations")

flags.DEFINE_integer('seed', default=10,
                     help="seed")

flags.DEFINE_bool('only_corrects', default=False,
                  help="evaluate only on correctly predicted examples")

flags.DEFINE_bool('only_wrongs', default=False,
                  help="evaluate only on wrong predicted examples")

flags.DEFINE_bool('only_learned', default=False,
                  help="evaluate only learned examples")

flags.DEFINE_bool('only_target_abstracts', default=False,
                  help="only count same target abstracts as correct")

flags.DEFINE_bool('include_eos', default=False,
                  help="include eos on target or not")

flags.DEFINE_bool('load_accums', default=False,
                  help="load_accumulators")

flags.DEFINE_string('samples_from_exp', default=None,
                     help="exp json to read samples")

K_EVALS = (1, 3, 5, 10, 25)


@dataclass
class LayerConfig:
    layer_prefixes: Sequence[str]
    layer_weights: Sequence[float] = field(default_factory=list)
    index: Optional[int] = 0
    
    def __post_init__(self):
        if len(self.layer_weights) == 0:
            for prefix in self.layer_prefixes:
                self.layer_weights.append(1.0)


def get_tfexample_decoder_examples():
    """Return tf dataset parser for examples."""

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
        return data

    return _parse_data


def get_tfexample_decoder_abstracts():
    """Return tf dataset parser for abstracts."""

    feature_dict = {
        'inputs_pretokenized': tf.io.FixedLenFeature([], tf.string),
        'targets_pretokenized': tf.io.FixedLenFeature([], tf.string),
        'masked_uri': tf.io.FixedLenFeature([], tf.string),
        'page_uri': tf.io.FixedLenFeature([], tf.string),
        'masked_type': tf.io.FixedLenFeature([], tf.string),
        'facts': tf.io.FixedLenFeature([], tf.string),
        'sentence_uris': tf.io.FixedLenFeature([], tf.string),
       }

    def _parse_data(proto):
        data = tf.io.parse_single_example(proto, feature_dict)
        return data  # (data['inputs_pretokenized'], data['targets_pretokenized'])

    return _parse_data


def load_dataset_from_tfrecord(dataset, load_fn):
    """Load one shard of a dataset from the dataset file."""
    ds_loader = dataset.map(load_fn()).as_numpy_iterator()
    return [d for d in ds_loader]


def tokenize(tokenizer: T5Tokenizer, record: Mapping):
    """Tokenize the inputs and targets of a record"""
    inputs = tokenizer(record['inputs_pretokenized'],
                       return_tensors='pt',
                       max_length=2048,
                       truncation=True,
                       ).input_ids
    targets = tokenizer(record['targets_pretokenized'],
                        return_tensors='pt').input_ids
   
    if not FLAGS.include_eos:
        targets = targets[:, :-1]
      
    return {'inputs': inputs, 'targets': targets[:, :-1]}


def sharded_open(filename: str):
    """Open sharded tfrecords as TFRecordDataset"""
    if '@' in filename:
        prefix, no_shards = filename.split('@')
        no_shards = int(no_shards)
        filenames = [f'{prefix}-{str(i).zfill(5)}-of-{str(no_shards).zfill(5)}' for i in range(no_shards)]
        dataset = tf.data.TFRecordDataset(filenames=filenames)
    else:
        dataset = tf.data.TFRecordDataset(filename)
    return dataset


def f_normalize(x):
    """Normalize vectors"""
    return F.normalize(x, dim=0)


def check_equal(a1: Mapping, a2: Mapping, collapse: bool):
    if collapse:
        return get_sentence(a1) == get_sentence(a2)
    else:
        return a1['sentence_uris'] == a2['sentence_uris']


def check_correct(a1: Mapping, fact_abstracts: Sequence[Mapping], collapse: bool):
    return any((check_equal(a1, a, collapse) for a in fact_abstracts))


def precision_recall(abstracts: Sequence[Mapping], fact_abstracts: Sequence[Mapping], ks: Sequence[int] = K_EVALS, collapse=False):
    """Calculate precision and recall given nearest ids and correct ids"""
    precision, recall = {}, {}
    for k in ks:
        nn_k = abstracts[:k]
        precision[k] = len([1 for a in nn_k if check_correct(a, fact_abstracts, collapse)]) / k
        recall[k] = len([1 for a in fact_abstracts if check_correct(a, nn_k, collapse)]) / len(fact_abstracts)
    return precision, recall


def reciprocal_rank(abstracts: Sequence[Mapping], fact_abstracts: Sequence[Mapping], collapse: bool = False):
    """Return reciprocal rank score"""
    for i, a in enumerate(abstracts):
        if check_correct(a, fact_abstracts, collapse):
            return 1 / (i+1)
    return 0


def get_gradients(model: MT5ForConditionalGeneration, data: Mapping):
    """Get Mapping[layer, gradient] given input and targets"""
    for param in model.parameters():
        param.grad = None

    model(input_ids=data['inputs'].cuda(model.cuda_no),
          labels=data['targets'].cuda(model.cuda_no)).loss.backward()
       
    if FLAGS.load_accums:
        grad = {("gradients." + name): param.grad.detach().flatten().div_(model.accums[name]) for name, param in model.named_parameters()}
    else:
        grad = {("gradients." + name): param.grad.detach().clone().flatten()  for name, param in model.named_parameters()}

    return grad


def get_activations(model: MT5ForConditionalGeneration, data: Mapping):
    """Get Mapping[layer, activation] given input and targets"""
    activations = {}
    with torch.no_grad():
        output = model(input_ids=data['inputs'].cuda(model.cuda_no),
                       labels=data['targets'].cuda(model.cuda_no),
                       output_hidden_states=True,
                       )
       
        for i, state in enumerate(output.encoder_hidden_states):
            activations[f'activations.encoder.block.{i}'] = state.mean(dim=1).squeeze()

        del output.encoder_hidden_states

        for i, state in enumerate(output.decoder_hidden_states):
            activations[f'activations.decoder.block.{i}'] = state.mean(dim=1).squeeze()

        del output

    return activations


def get_scores(vectors1, vectors2, f=lambda x: x):
    """Get dot product of dictionary of vectors with a preprocesser function f"""
    return {k: torch.dot(f(v), f(vectors2[k])).item() for k, v in vectors1.items()}


def get_all_scores_for_model(model, query, abstracts, encoder):
    """Get both cosine and uncosine scores for all the abstracts"""
    query_grad = encoder(model, query)
    scores = []
    nscores = []
    for i, abstract in enumerate(abstracts):
        abstract_grad = encoder(model, abstract)
        score = get_scores(query_grad, abstract_grad)
        scores.append(score)
        nscore = get_scores(query_grad, abstract_grad, f=f_normalize)
        nscores.append(nscore)
        del abstract_grad
    return scores, nscores


def merge_model_scores(scores):
    """Merge scores obtained from n checkpoints into single score by taking mean"""
    assert len(scores) > 0
    abstract_scores = []
    for j in range(len(scores[0])):
            abstract_scores.append(
                {k: np.mean([s[j][k] for s in scores])  # Mean over checkpoints
                                    for k, _ in scores[0][j].items()})
    return abstract_scores


def merge_new_scores_to_dict(scores, new_scores):
    """Accumulate new scores in to existing dictionary of scores"""
    if len(scores) == 0:
        return new_scores
    for score, new_score in zip(scores, new_scores):
        for (k, v) in new_score.items():
            assert k not in score
            score[k] = v
    return scores


def get_all_scores(models, tokenizer, query, abstracts):
    """Get both activation scores and mean gradient scores over checkpoints for list of models
       Note: We only take activation scores for last checkpoint.
    """
    query = tokenize(tokenizer, query)
    abstracts = [tokenize(tokenizer, a) for a in abstracts]
    all_scores = {'dot': {}, 'cosine': {}}
    
    for encoder in (get_gradients, get_activations):
        if encoder == get_activations:
            score, nscore = get_all_scores_for_model(models[-1], query, abstracts, encoder)
        else:
            scores = []
            nscores = []
            for model in models[:-1]:
                score, nscore = get_all_scores_for_model(model, query, abstracts, encoder)
                scores.append(score)
                nscores.append(nscore)
            score = merge_model_scores(scores)
            nscore = merge_model_scores(nscores)
       
        all_scores['dot'] = merge_new_scores_to_dict(all_scores['dot'], score)
        all_scores['cosine'] = merge_new_scores_to_dict(all_scores['cosine'], nscore)
   
    return all_scores


def get_sentence(abstract):
    targets = abstract['targets_pretokenized'].replace('<extra_id_0> ', '').strip()
    sentence = abstract['inputs_pretokenized'].replace('<extra_id_0>', targets)
    return sentence
    

def collapse_abstracts_and_scores(scores: Sequence[float], abstracts: Sequence[Mapping]):
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
    

def rerank_with_scores(abstracts: Sequence[Mapping], layer_scores: Mapping, layers: Optional[LayerConfig] = None, collapse: bool = False, normalize: bool = False):
    """Given layers prefixes we sum scores of these layers and rerank the abstracts"""
    abstract_scores = []
    if layers is not None:
        # Assuming our layer configurations are prefix codes
        def findindex(key):
            return np.where([key.startswith(layer)
                             for layer in layers.layer_prefixes])[0]
        
        sum_names = [weight_name for weight_name in layer_scores[0].keys() if len(findindex(weight_name)) > 0]
        w_weights = {weight_name: layers.layer_weights[findindex(weight_name)[0]] for weight_name in sum_names}
    else:
        sum_names = list(layer_scores[0].keys())
        w_weights = {k: 1.0 for k in sum_names}

    for layer_score in layer_scores:
        abstract_scores.append(np.sum([layer_score[k] * w_weights[k] for k in sum_names]))

    scores = np.array(abstract_scores)
    # merge abstracts and scores here
    if collapse:
        scores, abstracts = collapse_abstracts_and_scores(scores, abstracts)
    
    sorted_idxs = np.argsort(-scores)
    abstracts_reranked = [abstracts[i] for i in sorted_idxs]
    scores_reranked = scores[sorted_idxs]
    
    return abstracts_reranked, scores_reranked


def evaluate(example, abstracts, fact_abstracts, only_target_abstracts=False, collapse=False):
    """Evaluate nearast abstracts to get the metrics"""
    assert not (only_target_abstracts and collapse)
    
    if only_target_abstracts:
        fact_abstracts = list(filter(lambda a: a['targets_pretokenized'] == example['targets_pretokenized'], fact_abstracts))
        
    if collapse:
        identifier = get_sentence
        _, idxs = np.unique(list(map(identifier, fact_abstracts)), return_index=True)
        fact_abstracts = [fact_abstracts[id] for id in idxs]

    if len(fact_abstracts) == 0:
        logging.warning(f"empty fact abstract for query: {example}")
        return None, None, None

    # nn_ids = [a['page_uri'] for a in abstracts]
    precision, recall = precision_recall(abstracts, fact_abstracts, ks=K_EVALS, collapse=collapse)
    rr = reciprocal_rank(abstracts, fact_abstracts, collapse=collapse)
    return precision, recall, rr


def average_metrics(results):
    """Average the metrics over samples"""
    metrics = {'precision': {}, 'recall': {}}
    for k in K_EVALS:
        metrics['precision'][k] = np.mean([res['precision'][k] for res in results])
        metrics['recall'][k] = np.mean([res['recall'][k] for res in results])
    metrics['mrr'] = np.mean([res['rr'] for res in results])
    metrics['samples'] = results
    return metrics


def get_all_layer_configs(num_layers=12, exp_type="layers"):
    """Returns configurations listed below"""
    if exp_type == "layers":
        layer_configs = [LayerConfig(('gradients.shared',)), LayerConfig(('gradients.',))]
        layer_configs += [LayerConfig((f'gradients.encoder.block.{i}', f'gradients.decoder.block.{i}')) for i in range(num_layers)]
        layer_configs += [LayerConfig(('gradients.shared', f'gradients.encoder.block.{i}', f'gradients.decoder.block.{i}')) for i in range(num_layers)]
        layer_configs += [LayerConfig((f'gradients.encoder.block.{i}', )) for i in range(num_layers)]
        layer_configs += [LayerConfig(('gradients.shared', f'gradients.encoder.block.{i}')) for i in range(num_layers)]
        layer_configs += [LayerConfig((f'activations.encoder.block.{i}', f'activations.decoder.block.{i}')) for i in range(num_layers + 1)]
        layer_configs += [LayerConfig((f'activations.encoder.block.{i}',)) for i in range(num_layers + 1)]
        layer_configs += [LayerConfig(('activations.encoder.block.0', 'activations.decoder.block.0', f'activations.encoder.block.{i}', f'activations.decoder.block.{i}')) for i in range(1, num_layers+1)]
        layer_configs += [LayerConfig(('activations.encoder.block.0', f'activations.encoder.block.{i}', f'activations.decoder.block.{i}')) for i in range(1, num_layers+1)]
        layer_configs.append(LayerConfig(('activations.', )))
        layer_configs.append(LayerConfig(('activations.', 'gradients.')))
        layer_configs.append(LayerConfig(('activations.encoder.block.0', 'activations.decoder.block.0', 'gradients.shared')))
        layer_configs.append(LayerConfig(('activations.encoder.block.0', 'gradients.shared')))
    else:
        layer_configs = []
        index = 0
        for a in np.linspace(-5, 5, num=10):
            for b in np.linspace(-5, 5, num=10):
                layer_configs.append(LayerConfig(('activations.encoder.block.0', 'activations.decoder.block.0', 'gradients.shared'),
                                                  [a, b, 1.0],
                                                  index=index))
                index += 1
        for a in (0.0, 1.0):
            for b in (0.0, 1.0):
                layer_configs.append(LayerConfig(('activations.encoder.block.0', 'activations.decoder.block.0', 'gradients.shared'),
                                                  [a, b, 1.0],
                                                  index=index))
                index += 1
    return layer_configs

def run_all_layer_configs(models, tokenizer: T5Tokenizer, hashmap, samples, num_layers=12, exp_type="layers"):  # TODO: Read num_layers from models
    """Runs reranking experiments for all configurations listed below and returns the results"""
    layer_configs = get_all_layer_configs(num_layers, exp_type)
    
    results = {'cosine': {}, 'dot': {}}

    for (i, sample) in tqdm(enumerate(samples)):
        
        query = sample['example']
        
        abstracts = sample['nn_abstracts'] + sample['fact_abstracts'] + sample['distractors']
        
        random.shuffle(abstracts)
        
        # There might be intersecting abstracts in nn_abstracts and fact_abstracts and distractors
        identifier = lambda x: x['inputs_pretokenized'] + x['targets_pretokenized']
        _, indices = np.unique(list(map(identifier, abstracts)), return_index=True)
        abstracts = [abstracts[index] for index in indices]
        
        # Get similarity scores for all individual weights x {activations, gradients, both}
        scores = get_all_scores(models, tokenizer, query, abstracts)

        for k, result in results.items():  # cosine or dot
            
            for method in ('collapse', 'full'):  # eval methods
                
                if method not in result:
                    result[method] = {}
                    
                collapse = (method == 'collapse')
                only_target_abstracts = (method == 'target_abstracts')
                                      
                for config in layer_configs:
        
                    config_name = ",".join(config.layer_prefixes)
                    
                    if exp_type != "layers":
                        config_name = config_name + "_" + str(config.index)
                        
                    if config_name not in result[method]:
                        result[method][config_name] = []
            
                    abstracts_config, scores_config = rerank_with_scores(abstracts, scores[k], layers=config, collapse=collapse)
                               
                    precision, recall, rr = evaluate(query, abstracts_config, sample['fact_abstracts'], only_target_abstracts=only_target_abstracts, collapse=collapse)

                    if precision is not None:
                        result[method][config_name].append({
                            "example": sample["example"],
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
                print(config_name, "\t", method_name, '\t', k, '\t', average_result['mrr'])

    return metrics


def trim(output):
    """Trim the outputs for the accuracy evaluation."""
    output = output.replace('<extra_id_0>', '')
    index = output.find('<extra_id_1>')
    if index != -1:
        output = output[:index]
    output = output.strip()
    if len(output) > 0:
        if output[-1] == '.':
            output = output[:-1]
    return output.lower()
 
  
def run_random_baseline(samples):
    metrics = {}
    for sample in samples:
        query = sample['example']
        target = query['targets_pretokenized']
        abstracts = sample['nn_abstracts'] + sample['fact_abstracts'] + sample['distractors']
        random.shuffle(abstracts)
        target_abstracts = [a for a in abstracts if a['targets_pretokenized'] == target]
        other_abstracts = [a for a in abstracts if a['targets_pretokenized'] != target]
        abstracts_reranked = target_abstracts + other_abstracts
        scores_reranked = [1 for i in range(len(target_abstracts))] 
        scores_reranked += [0 for i in range(len(other_abstracts))]
        fact_abstracts = sample['fact_abstracts']
        
        for method in ('collapse', 'target_abstracts', 'full'):
                     
            collapse = (method == 'collapse')
            only_target_abstracts = (method == 'target_abstracts')    
            
            if method not in metrics:
                metrics[method] = []
        
            results = metrics[method]
            
            current_scores, current_abstracts = scores_reranked, abstracts_reranked
            
            if collapse:
                current_scores, current_abstracts = collapse_abstracts_and_scores(current_scores, current_abstracts)
            
            precision, recall, rr = evaluate(query, current_abstracts, fact_abstracts, only_target_abstracts=only_target_abstracts, collapse=collapse)
            
            if precision is not None:
                results.append({
                            "example": sample["example"],
                            "precision": precision,
                            "recall": recall,
                            "rr": rr,
                            "nn_abstracts": abstracts_reranked[:100],
                            "nn_scores": scores_reranked[:100]
                        })
                
    for method in metrics.keys():
        metrics[method] = average_metrics(metrics[method])
    return metrics


def rerun_baseline(samples):
    metrics = {}
    for method in ('collapse', 'target_abstracts', 'full'):
        if method not in metrics:
            metrics[method] = []
            
        collapse = (method == 'collapse')
        only_target_abstracts = (method == 'target_abstracts')
        
        for sample in copy.deepcopy(samples):
            scores, abstracts = (sample['nn']['scores'], sample['nn_abstracts'])
            
            if collapse:
                scores, abstracts = collapse_abstracts_and_scores(scores, abstracts)
                
            precision, recall, rr = evaluate(sample['example'], abstracts, sample['fact_abstracts'], only_target_abstracts=only_target_abstracts, collapse=collapse)
        
            if precision is None:
                continue
            
            if sample['rr'] != rr:
                logging.info(f"original scores are changed -- probably due to a modification in evaluation -- method: {method}")
                logging.info(f"example: {sample['example']}")
                logging.info(f"original ones: {(sample['precision'], sample['recall'], sample['rr'])}")
                logging.info(f"new ones: {(precision, recall, rr)}")
                
            sample['precision'] = precision
            sample['recall'] = recall
            sample['rr'] = rr
            
            metrics[method].append(sample)
                        
        logging.info(f"Samples filtered: {len(samples) - len(metrics[method])}")
        
        metrics[method] = average_metrics(metrics[method])
        
    return metrics
       
    
def get_model_accuracy(model, tokenizer: T5Tokenizer, samples, beam_size=3):
    """Get prediction labels for the given samples"""
    labels = []
    for sample in samples:
        raw_input = sample["example"]['inputs_pretokenized']
        data = tokenize(tokenizer, sample['example'])
        inputs = data['inputs']
        target = trim(sample['example']['targets_pretokenized'])
        outputs = model.generate(input_ids=inputs.cuda(model.cuda_no),
                                 num_beams=beam_size,
                                 num_return_sequences=3,
                                 max_length=20)
       
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs = tuple(map(trim, outputs))
        labels.append(target in outputs)
        print(f"Inputs: {raw_input} Target: {target}, Outputs: {outputs}")
    return np.array(labels)


EPS=1e-7


def main(_):
    for attr, flag_obj in FLAGS.__flags.items():
        print("--%s=%s" % (attr, flag_obj.value))
       
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    original_result = json.load(open(FLAGS.metrics_file))
    hashmap = json.load(open(FLAGS.hashmap_file, 'r'))
    tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")

    checkpoint_folders = FLAGS.checkpoint_folders.split(',')
    models = []
    for i, folder in enumerate(checkpoint_folders):
        model = MT5ForConditionalGeneration.from_pretrained(
                                        folder,
                                        local_files_only=True).cuda(i)
        
        if FLAGS.load_accums:
            print("loading_accums")
            accum =  MT5ForConditionalGeneration.from_pretrained(
                                        folder.replace("_model_", "_accum_"),
                                        local_files_only=True)
            model.accums = {}
            for (k, v) in accum.named_parameters():
                model.accums[k] = (torch.sqrt(v.data) + EPS).flatten().cuda(i)
                
        model.eval()
        model.cuda_no = i
        models.append(model)

    samples = original_result['samples']
    
    random.shuffle(samples)
    
    logging.info(f"Number of samples in original: {len(samples)}")
  
    labels = get_model_accuracy(models[-1],  # Last checkpoint is the best accuracy.
                                tokenizer,
                                samples,
                                beam_size=FLAGS.beam_size)
    

    
   
    logging.info(f"Mean accuracy of last checkpoint is {np.mean(labels)}")
   
    assert not (FLAGS.only_corrects and FLAGS.only_wrongs)
    
    if FLAGS.samples_from_exp is not None:
        with open(FLAGS.samples_from_exp) as f:
            exp_metrics = json.load(f)
        exp_inputs = [sample['example']['inputs_pretokenized'] 
                    for sample in exp_metrics['dot']['full']['bm25plus']['samples']]
        exp_uris = set(exp_inputs)
        samples = [sample for sample in samples if sample['example']['inputs_pretokenized'] in exp_uris]
        original_result['samples'] = samples
    else:
        if FLAGS.only_corrects:
            labels_zero = get_model_accuracy(models[0],
                                            tokenizer,
                                            samples,
                                            beam_size=FLAGS.beam_size)
            logging.info(f"Mean accuracy of first checkpoint is {np.mean(labels_zero)}")
            samples = [samples[i] for i in range(len(labels_zero)) if labels_zero[i]]
            original_result['samples'] = samples
            
        elif FLAGS.only_wrongs:
            samples = [samples[i] for i in range(len(labels)) if not labels[i]]
            original_result['samples'] = samples
            
        elif FLAGS.only_learned:    
            labels_zero = get_model_accuracy(models[0],
                                            tokenizer,
                                            samples,
                                            beam_size=FLAGS.beam_size)
            
            # labels_last = get_model_accuracy(models[-2],
            #                                  tokenizer,
            #                                  samples,
            #                                  beam_size=FLAGS.beam_size)
        
            # logging.info(f"Mean accuracy of last to second checkpoint is {np.mean(labels_last)}")
            samples = [samples[i] for i in range(len(labels)) if labels[i] and not labels_zero[i]]
            original_result['samples'] = samples
        
        if len(samples) > 100:
            samples = samples[:100]
            original_result['samples'] = samples    
    
    logging.info(f"Number of samples to evaluate is: {len(samples)}")
        
    logging.info(f"Original average scores: {(original_result['precision'], original_result['recall'], original_result['mrr'])}")
    
    baseline = rerun_baseline(samples)
    
    logging.info(f"Recalculated average scores: {(baseline['full']['precision'], baseline['full']['recall'], baseline['full']['mrr'])}")
    
    random_baseline = run_random_baseline(samples)

    metrics = run_all_layer_configs(models,
                                    tokenizer,
                                    hashmap,
                                    samples,
                                    exp_type=FLAGS.exp_type)
    
        # unnecessary memory usage, but makes my job easier in plotting
    for method in metrics['dot'].keys():
        metrics['dot'][method]['bm25plus'] = metrics['cosine'][method]['bm25plus'] = baseline[method]
        metrics['dot'][method]['random'] = metrics['cosine'][method]['random'] = random_baseline[method]
        
    
    output = FLAGS.output_metrics_prefix + ".json"
    with open(output, "w") as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    app.run(main)
    