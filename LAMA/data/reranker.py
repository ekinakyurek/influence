import json
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

flags.DEFINE_integer('beam_size', default=3,
                     help="beam size for accuracy calculations")

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

flags.DEFINE_bool('global_norm', default=False,
                  help="global normalization")


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


def tokenize(tokenizer, record):
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


def check_equal(a1, a2, collapse):
    if collapse:
        return a1['sentence_uris'] == a2['sentence_uris']
    else:
        return a1['sentence_uris'] == a2['sentence_uris'] and a1['targets_pretokenized'] == a2['targets_pretokenized']


def check_correct(a1, fact_abstracts, collapse):
    return any((check_equal(a1, a, collapse) for a in fact_abstracts))


def precision_recall(abstracts, fact_abstracts, ks=(1, 5, 10, 50, 100), collapse=False):
    """Calculate precision and recall given nearest ids and correct ids"""
    precision, recall = {}, {}
    for k in ks:
        nn_k = abstracts[:k]
        precision[k] = len([1 for a in nn_k if check_correct(a, fact_abstracts, collapse)]) / k
        recall[k] = len([1 for a in fact_abstracts if check_correct(a, nn_k, collapse)]) / len(fact_abstracts)
    return precision, recall


def reciprocal_rank(abstracts, fact_abstracts, collapse=False):
    """Return reciprocal rank score"""
    for i, a in enumerate(abstracts):
        if check_correct(a, fact_abstracts, collapse):
            return 1 / (i+1)
    return 0


def get_gradients(model, data):
    """Get Mapping[layer, gradient] given input and targets"""
    model.zero_grad()

    model(input_ids=data['inputs'].cuda(model.cuda_no),
          labels=data['targets'].cuda(model.cuda_no)).loss.backward()
   
    grad = {("gradients." + name): param.grad.detach().clone().flatten() for name, param in model.named_parameters()}

    return grad


def get_activations(model, data):
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

def get_score(v1, v2):
    score = torch.dot(v1, v2).item()
    norms = ((torch.linalg.norm(v1)**2).item(), (torch.linalg.norm(v2)**2).item())
    return (score, norms)

def get_scores(vectors1, vectors2):
    """Get dot product of dictionary of vectors with a preprocesser function f"""
    return {k: get_score(v, vectors2[k]) for k, v in vectors1.items()}


def get_all_scores_for_model(model, query, abstracts, encoder):
    """Get both cosine and uncosine scores for all the abstracts"""
    query_grad = encoder(model, query)
    scores = []
    for i, abstract in enumerate(abstracts):
        abstract_grad = encoder(model, abstract)
        score = get_scores(query_grad, abstract_grad)
        scores.append(score)
        del abstract_grad
    return scores


def merge_model_scores(scores):
    """Merge scores obtained from n checkpoints into single score by taking mean"""
    assert len(scores) > 0
    abstract_scores = []
    for j in range(len(scores[0])):
        abstract_scores.append(
            {k: (np.sum([s[j][k][0] for s in scores]),
                 (np.sum([s[j][k][1][0] for s in scores]),
                  np.sum([s[j][k][1][1] for s in scores])))
             for k, _ in scores[0][j].items()}
        )
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
    all_scores = {}
    
    for encoder in (get_gradients, get_activations):
        if encoder == get_activations:
            score = get_all_scores_for_model(models[-1],
                                             query,
                                             abstracts,
                                             encoder)
        else:
            scores = []
            for model in models[:-1]:
                score = get_all_scores_for_model(model,
                                                 query,
                                                 abstracts,
                                                 encoder)
                scores.append(score)
            score = merge_model_scores(scores)
       
        all_scores = merge_new_scores_to_dict(all_scores, score)
   
    return all_scores


def collapse_abstracts_and_scores(scores, abstracts):
    uri_to_indices = {}
    for i, a in enumerate(abstracts):
        uri = a['sentence_uris']
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
    

def rerank_with_scores(abstracts, layer_scores, layers=None, collapse=False, normalize=False):
    """Given layers prefixes we sum scores of these layers and rerank the abstracts"""
    abstract_scores = []
    if layers is not None:
        # Assuming our layer configurations are prefix codes
        inc = lambda key: any(key.startswith(layer) for layer in layers)
        sumk = [key for key in layer_scores[0].keys() if inc(key)]
    else:
        sumk = layer_scores[0].keys()

    for layer_score in layer_scores:
        value = np.sum([layer_score[k][0] for k in sumk])
        if normalize:
            norm1 = np.sum([layer_score[k][1][0] for k in sumk])
            norm2 = np.sum([layer_score[k][1][1] for k in sumk])
            norm = np.sqrt(norm1) * np.sqrt(norm2)
            value = value / norm
            
        abstract_scores.append(value)

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
        identifier = lambda x: x['sentence_uris']
        _, idxs = np.unique(list(map(identifier, fact_abstracts)), return_index=True)
        fact_abstracts = [fact_abstracts[id] for id in idxs]

    if len(fact_abstracts) == 0:
        logging.warning(f"empty fact abstract for query: {example}")
        return None, None, None

    # nn_ids = [a['page_uri'] for a in abstracts]
    precision, recall = precision_recall(abstracts, fact_abstracts, ks=(1, 5, 10, 50, 100), collapse=collapse)
    rr = reciprocal_rank(abstracts, fact_abstracts, collapse=collapse)
    return precision, recall, rr


def average_metrics(results):
    """Average the metrics over samples"""
    metrics = {'precision': {}, 'recall': {}}
    for k in (1, 5, 10,  50, 100):
        metrics['precision'][k] = np.mean([res['precision'][k] for res in results])
        metrics['recall'][k] = np.mean([res['recall'][k] for res in results])
    metrics['mrr'] = np.mean([res['rr'] for res in results])
    metrics['samples'] = results
    return metrics


def run_all_layer_configs(models, tokenizer: T5Tokenizer, hashmap, samples, num_layers=12):  # TODO: Read num_layers from models
    """Runs reranking experiments for all configurations listed below and returns the results"""
    layer_configs = [('gradients.shared',), ('gradients.',)]
    layer_configs += [(f'gradients.encoder.block.{i}', f'gradients.decoder.block.{i}') for i in range(num_layers)]
    layer_configs += [('gradients.shared', f'gradients.encoder.block.{i}', f'gradients.decoder.block.{i}') for i in range(num_layers)]
    layer_configs += [(f'activations.encoder.block.{i}', f'activations.decoder.block.{i}') for i in range(num_layers + 1)]
    layer_configs += [('activations.encoder.block.0', 'activations.decoder.block.0', f'activations.encoder.block.{i}', f'activations.decoder.block.{i}') for i in range(1, num_layers+1)]
    layer_configs.append(('activations.', ))
    layer_configs.append(('activations.', 'gradients.'))
    layer_configs.append(('activations.encoder.block.0', 'activations.decoder.block.0', 'gradients.shared'))

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
            
            for method in ('collapse', 'target_abstracts', 'full'):  # eval methods
                
                if method not in result:
                    result[method] = {}
                    
                collapse = (method == 'collapse')
                only_target_abstracts = (method == 'target_abstracts')
                                      
                for config in layer_configs:
        
                    config_name = ",".join(config)
                    
                    if config_name not in result[method]:
                        result[method][config_name] = []
            
                    abstracts_config, scores_config = rerank_with_scores(abstracts, scores, layers=config, collapse=collapse, normalize = k=='cosine')
                               
                    precision, recall, rr = evaluate(query, abstracts_config, sample['fact_abstracts'], only_target_abstracts=only_target_abstracts, collapse=collapse)

                    if precision is not None:
                        result[method][config_name].append({
                            "example": sample["example"],
                            "precision": precision,
                            "recall": recall,
                            "rr": rr,
                            "nn_abstracts": abstracts_config[:100],
                            "nn_scores": scores_config[:100].tolist(),
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


def main(_):
    for attr, flag_obj in FLAGS.__flags.items():
        print("--%s=%s" % (attr, flag_obj.value))
       
    np.random.seed(10)
    random.seed(10)

    original_result = json.load(open(FLAGS.metrics_file))
    hashmap = json.load(open(FLAGS.hashmap_file, 'r'))
    tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")

    checkpoint_folders = FLAGS.checkpoint_folders.split(',')
    models = []
    for i, folder in enumerate(checkpoint_folders):
        model = MT5ForConditionalGeneration.from_pretrained(
                                        folder,
                                        local_files_only=True).cuda(i)
        model.eval()
        model.cuda_no = i
        models.append(model)

    samples = original_result['samples']
  
    labels = get_model_accuracy(models[-1],  # Last checkpoint is the best accuracy.
                                tokenizer,
                                samples,
                                beam_size=FLAGS.beam_size)
    

    
   
    logging.info(f"Mean accuracy of last checkpoint is {np.mean(labels)}")
   
    assert not (FLAGS.only_corrects and FLAGS.only_wrongs)
    
    if FLAGS.only_corrects:
        samples = [samples[i] for i in range(len(labels)) if labels[i]]
        original_result['samples'] = samples
        
    elif FLAGS.only_wrongs:
        samples = [samples[i] for i in range(len(labels)) if not labels[i]]
        original_result['samples'] = samples
        
    elif FLAGS.only_learned:    
        labels_zero = get_model_accuracy(models[0],  # Last checkpoint is the best accuracy.
                                         tokenizer,
                                         samples,
                                         beam_size=FLAGS.beam_size)
    
        samples = [samples[i] for i in range(len(labels)) if labels[i] and not labels_zero[i]]
        original_result['samples'] = samples
        
        
    logging.info(f"Original average scores: {(original_result['precision'], original_result['recall'], original_result['mrr'])}")
    
    baseline = rerun_baseline(samples)
    
    logging.info(f"Recalculated average scores: {(baseline['full']['precision'], baseline['full']['recall'], baseline['full']['mrr'])}")
    
    random_baseline = run_random_baseline(samples)

    metrics = run_all_layer_configs(models,
                                    tokenizer,
                                    hashmap,
                                    samples)
    
        # unnecessary memory usage, but makes my job easier in plotting
    for method in metrics['dot'].keys():
        metrics['dot'][method]['bm25plus'] = metrics['cosine'][method]['bm25plus'] = baseline[method]
        metrics['dot'][method]['random'] = metrics['cosine'][method]['random'] = random_baseline[method]
        
    
    output = FLAGS.output_metrics_prefix + ".json"
    with open(output, "w") as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    app.run(main)
