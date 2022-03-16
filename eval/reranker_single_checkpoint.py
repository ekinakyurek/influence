import pathlib
import gzip
import os
import pickle
from typing import Mapping
import torch  # TODO(ekina): make this jax
from absl import app
from absl import flags
from absl import logging
import numpy as np
from tqdm import tqdm
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from transformers import AutoTokenizer
import random


FLAGS = flags.FLAGS


flags.DEFINE_string('metrics_file', default=None,
                    help='input metrics file that stores baseline statistics '
                         'and (examples, nn abstracts)')

flags.DEFINE_string('output_metrics_prefix', default=None,
                    help='output file for experiment results')

flags.DEFINE_string('checkpoint_folder', default=None,
                    help='checkpoint path to evaluate')

flags.DEFINE_integer('seed', default=10, help="seed")

flags.DEFINE_integer('gpu', default=0, help="id and gpu to use")

flags.DEFINE_bool('include_eos', default=False,
                  help="include eos on target or not")

flags.DEFINE_bool('load_accums', default=False,
                  help="load_accumulators")

flags.DEFINE_bool('calculate_activation_scores', default=False,
                  help="whether to calculate activation scores")

flags.DEFINE_bool('calculate_gradient_scores', default=False,
                  help="whether to calculate gradient scores")

flags.DEFINE_bool('use_entity_locations', default=False,
                  help="use only entity locations when calculating activation embeddings")


def _find_entity_locations(input, target, surface, name):
    output = [None, None]
    try:
        obj_start = input.index(surface)
        obj_end = min(obj_start + len(surface), len(input))
        output[0] = (obj_start, obj_end, name)
    except ValueError:
        logging.warning("No object in query")

    try:
        obj_start = target.index(surface)
        obj_end = min(obj_start + len(surface), len(target))
        output[1] = (obj_start, obj_end, name)
    except ValueError:
        logging.warning("No object in query")

    return output


def find_entity_locations(data: Mapping):
    obj_surface = data['obj_surface']
    sub_surface = data['sub_surface']
    input = data['inputs_pretokenized']
    target = data['targets_pretokenized']

    if obj_surface is not None:
        obj_loc_in_input, obj_loc_in_target = _find_entity_locations(
                                                    input,
                                                    target,
                                                    obj_surface,
                                                    "object")

    if sub_surface is not None:
        sub_loc_in_input, sub_loc_in_target = _find_entity_locations(
                                                    input,
                                                    target,
                                                    sub_surface,
                                                    "subject")

    return ((obj_loc_in_input, sub_loc_in_input),
            (obj_loc_in_target, sub_loc_in_target))


def tokenize(tokenizer: T5Tokenizer,
             record: Mapping):
    """Tokenize the inputs and targets of a record"""
    tokenized_inputs = tokenizer(
                       record['inputs_pretokenized'],
                       return_tensors='pt',
                       max_length=2048,
                       return_offsets_mapping=True,
                       truncation=True,)

    inputs = tokenized_inputs.input_ids

    tokenized_targets = tokenizer(record['targets_pretokenized'],
                                  return_tensors='pt',
                                  return_offsets_mapping=True)

    targets = tokenized_targets.input_ids

    if not FLAGS.include_eos:
        targets = targets[:, :-1]

    # Fixme: it seems like we discard more than eos
    # But this was the scores in the paper
    output = {'inputs': inputs, 'targets': targets[:, :-1]}

    if FLAGS.use_entity_locations and 'obj_surface' in record:
        entity_locations = find_entity_locations(record)
        entity_indices = find_entity_indices(tokenizer,
                                             entity_locations,
                                             tokenized_inputs,
                                             tokenized_targets)
        output['entity_indices'] = entity_indices

    return output


def find_entity_indices_single(tokenizer, location, tokenized_input):
    output = [tokenized_input.char_to_token(location[0]),
              tokenized_input.char_to_token(location[1]),
              location[2]]
    # if start position is None, the answer passage has been truncated
    if output[0] is None:
        output[0] = tokenizer.model_max_length

    # if end position is None, the 'char_to_token' function points to the space before the correct token - > add + 1
    if output[1] is None:
        output[1] = tokenized_input.char_to_token(location[1] + 1)
        if output[1] is None:
            output[1] = output[0]
    return output


def find_entity_indices(tokenizer, entity_locations, tokenized_input, tokenized_target):
    input_positions = []
    for i, location in enumerate(entity_locations[0]):
        if location is None:
            continue
        if location[0] is None or location[1] is None:
            continue
        input_positions.append(find_entity_indices_single(tokenizer, location, tokenized_input))

    target_positions = []
    for i, location in enumerate(entity_locations[1]):
        if location is None:
            continue
        if location[0] is None or location[1] is None:
            continue
        target_positions.append(find_entity_indices_single(tokenizer, location, tokenized_target))

    return (input_positions, target_positions)


def get_gradients(model: MT5ForConditionalGeneration, data: Mapping):
    """Get Mapping[layer, gradient] given input and targets"""
    for param in model.parameters():
        param.grad = None

    model(input_ids=data['inputs'].cuda(model.cuda_no),
          labels=data['targets'].cuda(model.cuda_no)).loss.backward()

    if FLAGS.load_accums:
        grad = {("gradients." + name): param.grad.detach().flatten().div_(model.accums[name])
                for name, param in model.named_parameters()}
    else:
        grad = {("gradients." + name): param.grad.detach().clone().flatten()
                for name, param in model.named_parameters()}

    return grad


def get_activations(model: MT5ForConditionalGeneration, data: Mapping):
    """Get Mapping[layer, activation] given input and targets"""
    activations = {}
    with torch.no_grad():
        output = model(input_ids=data['inputs'].cuda(model.cuda_no), labels=data['targets'].cuda(model.cuda_no), output_hidden_states=True)

        if 'entity_indices' in data:
            input_indices, output_indices = data['entity_indices']
        else:
            input_indices, output_indices = [], []

        for i, state in enumerate(output.encoder_hidden_states):
            if input_indices:
                activations[f'activations.encoder.block.{i}'] = 0
                for location in input_indices:
                    if location[1] < state.size(1):
                        activations[f'activations.encoder.block.{i}'] += state[:, location[1], :].squeeze()
                    else:
                        activations[f'activations.encoder.block.{i}'] += state[:, -1, :].squeeze()
            else:
                activations[f'activations.encoder.block.{i}'] = state.mean(dim=1).squeeze()

        del output.encoder_hidden_states

        for i, state in enumerate(output.decoder_hidden_states):
            if output_indices:
                activations[f'activations.decoder.block.{i}'] = 0
                for location in output_indices:
                    if location[1] < state.size(1):
                        activations[f'activations.decoder.block.{i}'] += state[:, location[1], :].squeeze()
                    else:
                        activations[f'activations.decoder.block.{i}'] += state[:, -1, :].squeeze()
            else:
                activations[f'activations.decoder.block.{i}'] = state.mean(dim=1).squeeze()

        del output

    return activations


def get_score(v1: torch.Tensor,
              v2: torch.Tensor,
              f=lambda x: x,
              eps: float = 1e-12):
    score = torch.dot(v1, v2).item()
    norms = ((torch.linalg.norm(v1)**2).clamp_min(eps).item(),
             (torch.linalg.norm(v2)**2).clamp_min(eps).item())
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


def merge_new_scores_to_dict(scores, new_scores):
    """Accumulate new scores in to existing dictionary of scores"""
    if len(scores) == 0:
        return new_scores
    for score, new_score in zip(scores, new_scores):
        for (k, v) in new_score.items():
            assert k not in score
            score[k] = v
    return scores


def get_all_scores(model, tokenizer, query, abstracts):
    """Get both activation scores and mean gradient scores over checkpoints for list of models
       Note: We only take activation scores for last checkpoint.
    """
    query = tokenize(tokenizer, query)
    abstracts = [tokenize(tokenizer, a) for a in abstracts]
    all_scores = []
    encoders = []

    if FLAGS.calculate_activation_scores:
        encoders.append(get_activations)

    if FLAGS.calculate_gradient_scores:
        encoders.append(get_gradients)

    for encoder in encoders:
        if encoder == get_activations:
            score = get_all_scores_for_model(model,
                                             query,
                                             abstracts,
                                             encoder)
        else:
            score = get_all_scores_for_model(model,
                                             query,
                                             abstracts,
                                             encoder)

        all_scores = merge_new_scores_to_dict(all_scores, score)

    return all_scores


def get_sentence(abstract):
    targets = abstract['targets_pretokenized'].replace('<extra_id_0> ', '').strip()
    sentence = abstract['inputs_pretokenized'].replace('<extra_id_0>', targets)
    return sentence


def identifier(x):
    return x['inputs_pretokenized'] + x['targets_pretokenized']


def run_all_query_scores(model, score_fn, tokenizer: T5Tokenizer, samples):
    """Runs reranking experiments for all configurations listed below and returns the results"""
    query_scores = []
    for (index, sample) in tqdm(enumerate(samples)):
        query = sample['example']

        abstracts = sample['nn_abstracts'] + sample['fact_abstracts'] + sample['distractors']
        rng = np.random.default_rng(0)
        rng.shuffle(abstracts)
        _, indices = np.unique(list(map(identifier, abstracts)),
                               return_index=True)
        abstracts = [abstracts[ind] for ind in indices]

        # Get similarity scores for all individual weights x {activations, gradients, both}
        scores = score_fn(model, tokenizer, query, abstracts)
        query_scores.append(scores)
    return query_scores


def main(_, score_fn=get_all_scores):
    for attr, flag_obj in FLAGS.__flags.items():
        logging.info("--%s=%s" % (attr, flag_obj.value))

    assert FLAGS.calculate_activation_scores or FLAGS.calculate_gradient_scores

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    with gzip.open(FLAGS.metrics_file, 'rb') as handle:
        original_result = pickle.load(handle)

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")

    checkpoint_folder = FLAGS.checkpoint_folder

    model = MT5ForConditionalGeneration.from_pretrained(
                                    checkpoint_folder,
                                    local_files_only=True).cuda(FLAGS.gpu)

    checkpoint_name = pathlib.PurePath(checkpoint_folder).name

    if FLAGS.load_accums:
        logging.info("loading accumulators")
        accum = MT5ForConditionalGeneration.from_pretrained(
                                    checkpoint_folder.replace("_model_",
                                                              "_accum_"),
                                    local_files_only=True)
        model.accums = {}
        for (k, v) in accum.named_parameters():
            model.accums[k] = (torch.sqrt(v.data) + 1e-7).flatten()\
                                                         .cuda(FLAGS.gpu)

    model.eval()
    model.cuda_no = FLAGS.gpu

    samples = original_result['samples']

    logging.info(f"Number of samples in original: {len(samples)}")

    baseline = original_result['evals']['bm25plus']['collapse']
    baseline_results = (baseline['precision'],
                        baseline['recall'],
                        baseline['mrr'])

    logging.info(f"Original average scores: "
                 f"{baseline_results}")

    scores = run_all_query_scores(model,
                                  score_fn,
                                  tokenizer,
                                  samples)

    output = os.path.join(FLAGS.output_metrics_prefix,  f"{checkpoint_name}.pickle")

    with gzip.open(output, "wb") as f:
        pickle.dump(scores, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    app.run(main)
