import os
import json
import pdb
from absl import app
from absl import flags
from absl import logging
from os import listdir
import numpy as np
import tensorflow as tfp


FLAGS = flags.FLAGS
flags.DEFINE_string('hashmap_file', default=None,
                    help='input file path to convert')

flags.DEFINE_string('abstract_uri_list', default=None,
                    help='output file path to write hashmap')

flags.DEFINE_string('output_file', default=None,
                    help='output file to write')


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def dump_map_to_json(hashmap, output_file):
    with open(output_file, 'w') as f:
        json.dump(hashmap, f, cls=SetEncoder)

def main(argv):
    uri_list = np.array(
        open(FLAGS.abstract_uri_list, 'r').read().split('\n'))
    uri_list = set(uri_list[:-1])

    hashmap = json.load(open(FLAGS.hashmap_file, 'r'))

    for ids in hashmap.values():
        for i in range(len(ids)-1, -1, -1):
            if ids[i] not in uri_list:
                del ids[i]

    dump_map_to_json(hashmap, FLAGS.output_file)


if __name__ == '__main__':
    app.run(main)
