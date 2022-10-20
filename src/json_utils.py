import json
from typing import List, Mapping
import numpy as np


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def dump_map_to_json(hashmap: Mapping, output_file: str):
    with open(output_file, "w") as f:
        json.dump(hashmap, f, cls=SetEncoder)


def dump_abstracts_json(abstracts: List[Mapping], output_file: str):
    with open(output_file, "w") as f:
        for abstract in abstracts:
            json.dump(abstract, f)
            f.write("\n")


def load_synth_dataset(path):
    data = []
    with open(path) as f:
        data = [json.loads(line.strip()) for line in f]
    return np.array(data)
