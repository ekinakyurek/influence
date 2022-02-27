import json
from absl import app
from absl import flags
import numpy as np
import random

from src.tf_utils import load_examples_from_tf_records
from src.metric_utils import reciprocal_rank, precision_recall, K_EVALS


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


def main(_):
    uri_list = np.array(
        open(FLAGS.abstract_uri_list, 'r').read().split('\n'))
    uri_list = uri_list[:-1]

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
                abstracts.append(json.loads(line))
        assert ([a['page_uri'] for a in abstracts] == uri_list).all()
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
    for p, index in enumerate(indices):
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
                                             ks=(1, 5, 10, 50, 100))

        precisions.append(precision)
        recalls.append(recall)
        rrs.append(reciprocal_rank(nn_ids, uris))
        if len(metrics['samples']) < 100 and FLAGS.abstract_file is not None:
            facts_abstracts = []
            for uri in uris:
                ids_w_uri = np.where(uri_list == uri)[0]
                facts_abstracts.extend(abstracts[ids_w_uri])
            if len(facts_abstracts) == 0:
                continue

            distractors = [a for a in abstracts if a['targets_pretokenized'] == example['targets_pretokenized']][:100] + np.random.choice(abstracts, 100, replace=False).tolist()

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
    # print(metrics)

    with open(FLAGS.output_file, 'w') as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    app.run(main)
