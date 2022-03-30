import json
from absl import app
from absl import flags
from absl import logging
import numpy as np
import random
import pickle
import gzip
from tqdm import tqdm
from src.metric_utils import precision_recall
from src.metric_utils import reciprocal_rank
from src.metric_utils import K_EVALS
from src.tf_utils import load_examples_from_tf_records


FLAGS = flags.FLAGS

flags.DEFINE_string('nn_list_file', default=None,
                    help='nearest neighbors of the test examples in the same order')

flags.DEFINE_string('abstract_uri_list', default=None,
                    help='page uri list of abstracts in the same order with abstract vectors')

flags.DEFINE_string('abstract_file', default=None,
                    help='abstract file')

flags.DEFINE_string('test_data', default=None,
                    help='test examples to evaluate')

flags.DEFINE_string('hashmap_file', default=None,
                    help='hashmap that maps relation,obj,subj->page_uris')

flags.DEFINE_string('output_file', default=None,
                    help='output file path to writer neighbours')

flags.DEFINE_bool('target_only', default=False,
                  help='targets abstracts only')

flags.DEFINE_bool('only_masked_sentence', default=False,
                  help='convert abstracts to single sentence by extracting only the masked sentence')

flags.DEFINE_boolean('disable_tqdm', False,
                     help='Disable tqdm')


def extract_masked_sentence(abstract: str, term='<extra_id_0>'):
    term_start = abstract.find(term)
    assert term_start > -1
    term_end = term_start + len(term)
    sentence_start = abstract.rfind('. ', 0, term_start)
    if sentence_start == -1:
        sentence_start = 0
    else:
        sentence_start += 2
    sentence_end = abstract.find('. ', term_end)
    if sentence_end == -1:
        sentence_end = abstract.find('.', term_end)
    sentence_end = min(sentence_end + 1, len(abstract))
    return abstract[sentence_start:sentence_end]


def main(_):
    uri_list = np.array(
        open(FLAGS.abstract_uri_list, 'r').read().split('\n'))

    uri_list = uri_list[:-1]

    uri_to_ids = {}

    for i, uri in enumerate(uri_list):
        if uri not in uri_to_ids:
            uri_to_ids[uri] = [i]
        else:
            uri_to_ids[uri].append(i)

    logging.info(f"Length of abstracts {len(uri_list)}")

    hashmap = json.load(open(FLAGS.hashmap_file, 'r'))

    examples = load_examples_from_tf_records(FLAGS.test_data)

    nns = []
    with open(FLAGS.nn_list_file, 'r') as f:
        for line in f:
            nns.append(json.loads(line))

    if FLAGS.abstract_file is not None:
        abstracts = []
        with open(FLAGS.abstract_file, 'r') as f:
            for line in f:
                abstract = json.loads(line)
                if FLAGS.only_masked_sentence:
                    abstract['inputs_pretokenized'] = extract_masked_sentence(abstract['inputs_pretokenized'])
                abstracts.append(abstract)
        assert (np.array([a['sentence_uris'] for a in abstracts]) == uri_list).all()
        abstracts = np.array(abstracts)

    assert len(nns) == len(examples)
    assert max([max(nn['neighbor_ids']) for nn in nns]) < len(uri_list)

    metrics = {'precision': {}, 'recall': {}, 'samples': []}

    precisions = []
    recalls = []
    rrs = []
    np.random.seed(10)
    random.seed(10)
    indices = np.random.permutation(len(examples))
    counted = 0
    for p, index in enumerate(tqdm(indices, disable=FLAGS.disable_tqdm)):
        nn = nns[index]
        example = examples[index]
        nn_ids = uri_list[nn['neighbor_ids']]

        key = ",".join((example['predicate_id'],
                        example['obj_uri'],
                        example['sub_uri']))

        uris = hashmap.get(key, None)

        if uris is None or len(uris) == 0:
            continue
        counted += 1

        precision, recall = precision_recall(nn_ids,
                                             uris,
                                             ks=K_EVALS)
        rr = reciprocal_rank(nn_ids, uris)
        precisions.append(precision)
        recalls.append(recall)
        rrs.append(rr)

        if FLAGS.abstract_file is not None and len(metrics['samples']) < 10000:

            facts_abstracts = []
            for uri in uris:
                ids_w_uri = uri_to_ids[uri]
                facts_abstracts.extend(abstracts[ids_w_uri])

            if len(facts_abstracts) == 0:
                continue

            if FLAGS.target_only:
                distractors = [a for a in abstracts
                               if a['targets_pretokenized'] == example['targets_pretokenized']][:200]
            else:
                distractors = [a for a in abstracts
                               if a['targets_pretokenized'] == example['targets_pretokenized']][:100]

                distractors.extend(np.random.choice(abstracts,
                                                    100,
                                                    replace=False))

            metrics['samples'].append({"example": example,
                                       "precision": precision,
                                       "recall": recall,
                                       "rr": rrs[-1],
                                       "nn": nn,
                                       "nn_abstracts": abstracts[nn['neighbor_ids']].tolist(),
                                       "fact_abstracts": facts_abstracts,
                                       "distractors":  distractors})

    for k in K_EVALS:
        metrics['precision'][k] = np.mean([p[k] for p in precisions])
        metrics['recall'][k] = np.mean([r[k] for r in recalls])
    metrics['mrr'] = np.mean(rrs)

    with gzip.open(FLAGS.output_file, 'w') as f:
        pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    app.run(main)
