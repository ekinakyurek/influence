import json
import os
from absl import app, flags
from src.json_utils import dump_map_to_json


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "abstract_file", default=None, help="input file path to convert"
)

flags.DEFINE_string(
    "output_file", default=None, help="output file path to write hashmap"
)

flags.mark_flag_as_required("abstract_file")
flags.mark_flag_as_required("output_file")


def _get_all_facts_from_abstract(abstract):
    """Apply span masking to boundaries."""
    triples = abstract["triples"]
    facts = []
    for triple in triples:
        _, predicate = os.path.split(triple["predicate"]["uri"])
        _, subject = os.path.split(triple["subject"]["uri"])
        _, object = os.path.split(triple["object"]["uri"])
        if object.startswith("XML") or subject.startswith("XML"):
            continue
        fact = ",".join((predicate, object, subject))
        facts.append(fact)
    return facts


def main(argv):
    hashmap = {}
    with open(FLAGS.abstract_file, "r") as f:
        for i, line in enumerate(f):
            abstracts = json.loads(line)
            for abstract in abstracts:
                _, uri = os.path.split(abstract["uri"])
                facts = _get_all_facts_from_abstract(abstract)
                for fact in facts:
                    uris = hashmap.setdefault(fact, set())
                    uris.add(uri)
            dump_map_to_json(hashmap, FLAGS.output_file)

    return hashmap


if __name__ == "__main__":
    app.run(main)
