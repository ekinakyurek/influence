    import os
import json
import pdb
from absl import app
from absl import flags
from absl import logging
from os import listdir
import tensorflow as tf


FLAGS = flags.FLAGS
flags.DEFINE_string('abstract_file', default=None,
                    help='input file path to convert')

flags.DEFINE_string('output_file', default=None,
                    help='output file path to write hashmap')

flags.mark_flag_as_required('abstract_file')
flags.mark_flag_as_required('output_file')


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def _get_all_facts_from_abstract(abstract):
    """Apply span masking to boundaries."""
    triples = abstract['triples']
    facts = []
    for triple in triples:
        _, predicate = os.path.split(triple['predicate']['uri'])
        _, subject = os.path.split(triple['subject']['uri'])
        _, object = os.path.split(triple['object']['uri'])
        if object.startswith('XML') or subject.startswith('XML'):
            continue
        fact = ','.join((predicate, object, subject))
        facts.append(fact)
    return facts


def dump_map_to_json(hashmap, output_file):
    with open(output_file, 'w') as f:
        json.dump(hashmap, f, cls=SetEncoder)


def main(argv):
    abstracts_dict = {}
    hashmap = {}
    with open(FLAGS.abstract_file, 'r') as f:
        for i, line in enumerate(f):
            abstracts = json.loads(line)
            for abstract in abstracts:
                _, uri = os.path.split(abstract['uri'])
                facts = _get_all_facts_from_abstract(abstract)
                for fact in facts:
                    uris = hashmap.setdefault(fact, set())
                    uris.add(uri)
            dump_map_to_json(hashmap, FLAGS.output_file)

    return hashmap


if __name__ == '__main__':
    app.run(main)
