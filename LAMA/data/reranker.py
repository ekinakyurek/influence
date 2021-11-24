import json
import torch  # TODO(ekina): make this jax
import torch.nn.functional as F
from absl import app
from absl import flags
import numpy as np
from tqdm import tqdm
from transformers import MT5ForConditionalGeneration, T5Tokenizer
import random
import tensorflow as tf
from transformers.utils.dummy_pt_objects import EncoderDecoderModel, MT5Model

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
                    help='hashmap that maps relation,obj,subj->page_uris')

flags.DEFINE_integer('beam_size', default=3,
                  help="beam size for accuracy calculations")

flags.DEFINE_bool('only_corrects', default=False,
                  help="evaluate only on correctly predicted examples")

flags.DEFINE_bool('only_wrongs', default=False,
                  help="evaluate only on wrong predicted examples")

flags.DEFINE_bool('include_eos', default=False,
                  help="include eos on target or not")



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
                       return_tensors='pt').input_ids
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


def precision_recall(nearest_ids, correct_ids, ks=(1, 5, 10, 50, 100)):
    """Calculate precision and recall given nearest ids and correct ids"""
    precision, recall = {}, {}
    for k in ks:
        nn_k = nearest_ids[:k]
        precision[k] = len([1 for id in nn_k if id in correct_ids]) / k
        recall[k] = len([1 for id in correct_ids if id in nn_k]) / len(correct_ids)
    return precision, recall


def reciprocal_rank(nearest_ids, correct_ids):
    """Return reciprocal rank score"""
    for i, id in enumerate(nearest_ids):
        if id in correct_ids:
            return 1 / (i+1)
    return 0


def get_gradients(model, data):
    """Get Mapping[layer, gradient] given input and targets"""
    model.zero_grad()

    model(input_ids=data['inputs'].cuda(model.cuda_no),
          labels=data['targets'].cuda(model.cuda_no)).loss.backward()
   
    grad = {("gradients." + name): param.grad.detach().flatten() 
                                for name, param in model.named_parameters()}

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
            activations[f'activations.encoder.block.{i}'] = state.sum(dim=1).squeeze()

        del output.encoder_hidden_states

        for i, state in enumerate(output.decoder_hidden_states):
            activations[f'activations.decoder.block.{i}'] = state.sum(dim=1).squeeze()

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
    for i, abstract in tqdm(enumerate(abstracts)):
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
                {k: np.mean([s[j][k] for s in scores])  #  Mean over checkpoints
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


def rerank_with_scores(abstracts, layer_scores, layers=None):
    """Given layers prefixes we sum scores of these layers and rerank the abstracts"""
    abstract_scores = []
    if layers is not None:
        # Assuming our layer configurations are prefix codes
        inc = lambda key: any(key.startswith(layer) for layer in layers)
        sumk = [key for key in layer_scores[0].keys() if inc(key)]
    else:
        sumk = layer_scores[0].keys()

    for layer_score in layer_scores:
        abstract_scores.append(np.sum([layer_score[k] for k in sumk]))

    scores = np.array(abstract_scores)
    sorted_idxs = np.argsort(-scores)
    abstracts_reranked = [abstracts[i] for i in sorted_idxs]
    scores_reranked = scores[sorted_idxs]
    return abstracts_reranked, scores_reranked


def evaluate(example, abstracts, hashmap):
    """Evaluate nearast abstracts to get the metrics"""
    key = ",".join((example['predicate_id'],
                    example['obj_uri'],
                    example['sub_uri']))

    uris = hashmap.get(key, None)
    if uris is None or len(uris) == 0:
        return None, None, None

    nn_ids = [a['page_uri'] for a in abstracts]
    precision, recall = precision_recall(nn_ids, uris, ks=(1, 5, 10, 50, 100))
    rr = reciprocal_rank(nn_ids, uris)
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


def run_all_layer_configs(models, tokenizer: T5Tokenizer, hashmap, samples, num_layers=12):  #  TODO: Read num_layers from models
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

        for config in layer_configs:
            
            config_name = ",".join(config)
            
            for k, result in results.items():  # cosine or dot
                abstracts_config, scores_config = rerank_with_scores(abstracts, scores[k], layers=config)
                precision, recall, rr = evaluate(query, abstracts_config, hashmap)
                 
                if config_name not in result:
                    result[config_name] = []


                result[config_name].append({
                    "example": sample["example"],
                    "precision": precision,
                    "recall": recall,
                    "rr": rr,
                    "nn_abstracts": abstracts_config[:100],
                    "nn_scores": scores_config[:100].tolist(),
                })

    metrics = {'cosine': {}, 'dot': {}}
    for k, result in results.items():
        for (config_name, res) in result.items():
            metrics[k][config_name] = average_metrics(res)
            print(config_name, "\t", k, '\t', metrics[k][config_name]['mrr'])

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
 
  
def run_random_baseline(hashmap, samples):
    results = []
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
        
        precision, recall, rr = evaluate(query, abstracts_reranked, hashmap)
        
        results.append({
                    "example": sample["example"],
                    "precision": precision,
                    "recall": recall,
                    "rr": rr,
                    "nn_abstracts": abstracts_reranked[:100],
                    "nn_scores": scores_reranked[:100]
                })
    return average_metrics(results)
    
    
def get_model_accuracy(model: MT5Model, tokenizer: T5Tokenizer, samples, beam_size=3):
    """Get prediction labels for the given samples"""
    labels = []
    for sample in samples:
        raw_input =  sample["example"]['inputs_pretokenized']
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
   
    print(f"Mean accuracy of last checkpoint is {np.mean(labels)}")
   
    assert FLAGS.only_corrects != FLAGS.only_wrongs
    if FLAGS.only_corrects:
        samples = [samples[i] for i in range(len(labels)) if labels[i]]
        original_result['samples'] = samples
        
    if FLAGS.only_wrongs:
        samples = [samples[i] for i in range(len(labels)) if not labels[i]]
        original_result['samples'] = samples

    for sample in samples:
        precision, recall, rr = evaluate(sample['example'], sample['nn_abstracts'], hashmap)
        if sample['rr'] != rr:
            print("original scores are changed -- probably due to a modification in evaluation")
        sample['precision'] = precision
        sample['recall'] = recall
        sample['rr'] = rr
        
    original_result = average_metrics(samples)
       
    # base = FLAGS.output_metrics_prefix + "_base.json"
    # with open(base, "w") as f:
    #     json.dump(original_result, f)
        
    random_baseline = run_random_baseline(hashmap, samples)

    metrics = run_all_layer_configs(models,
                                    tokenizer,
                                    hashmap,
                                    samples)
    
    # unnecessary memory usage, but makes my job easier in plotting
    metrics['dot']['bm25plus'] = metrics['cosine']['bm25plus'] = original_result
    metrics['dot']['random'] = metrics['cosine']['random'] = random_baseline
    
    output = FLAGS.output_metrics_prefix + "_all.json"
    with open(output, "w") as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    app.run(main)
