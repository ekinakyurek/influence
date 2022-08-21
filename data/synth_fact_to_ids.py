import json
from absl import app, flags


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "test_file", default=None, help="test file to extract fact_to_ids_file"
)

flags.DEFINE_string(
    "fact_to_ids_output_file", default=None, help="fact to ids output file"
)


def main(argv):
    # write to json
    fact_to_ids = {}
    with open(FLAGS.test_file) as f:
        for line in f:
            data = json.loads(line.strip())
            fact_to_ids[data["uuid"]] = data["proponents"]

    with open(FLAGS.fact_to_ids_output_file, "w") as f:
        json.dump(fact_to_ids, f)


if __name__ == "__main__":
    app.run(main)
