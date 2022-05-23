import json
import numpy as np
from absl import app, flags
from src.json_utils import dump_map_to_json


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "hashmap_file", default=None, help="input file path to convert"
)

flags.DEFINE_string(
    "abstract_uri_list", default=None, help="output file path to write hashmap"
)

flags.DEFINE_string("output_file", default=None, help="output file to write")


def main(argv):
    uri_list = np.array(open(FLAGS.abstract_uri_list, "r").read().split("\n"))
    uri_list = set(uri_list[:-1])

    hashmap = json.load(open(FLAGS.hashmap_file, "r"))

    for ids in hashmap.values():
        for i in range(len(ids) - 1, -1, -1):
            if ids[i] not in uri_list:
                del ids[i]

    dump_map_to_json(hashmap, FLAGS.output_file)


if __name__ == "__main__":
    app.run(main)
