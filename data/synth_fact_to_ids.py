import json
from absl import app, flags


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "query_file", default=None, help="test file to extract fact_to_ids_file"
)

flags.DEFINE_string(
    "abstract_file", default=None, help="test file to extract fact_to_ids_file"
)

flags.DEFINE_string(
    "fact_to_ids_output_file", default=None, help="fact to ids output file"
)


def main(argv):
    # write to json
    fact_to_ids = {}
    with open(FLAGS.query_file) as f:
        for line in f:
            data = json.loads(line.strip())
            fact_to_ids[data["uuid"]] = data["proponents"]

    id_to_full_uuids = {}
    with open(FLAGS.abstract_file) as f:
        for line in f:
            abstract = json.loads(line.strip())
            a_uuid = abstract["uuid"]
            a_uuids = a_uuid.split(",")
            for uuid in a_uuids:
                if uuid in id_to_full_uuids:
                    id_to_full_uuids[uuid].append(a_uuid)
                else:
                    id_to_full_uuids[uuid] = [a_uuid]

    fact_to_full_uuids = {}
    for fact, uuids in fact_to_ids.items():
        fact_to_full_uuids[fact] = []
        for uuid in uuids:
            full_ids = id_to_full_uuids[uuid]
            fact_to_full_uuids[fact] += full_ids

    with open(FLAGS.fact_to_ids_output_file, "w") as f:
        json.dump(fact_to_full_uuids, f)


if __name__ == "__main__":
    app.run(main)
