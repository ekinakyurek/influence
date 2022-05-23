import json
from absl import app, flags
from src.tf_utils import _bytes_feature, tf


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_file", default=None, help="input file path to convert"
)

flags.DEFINE_integer("shard_no", default=1, help="shard number of current file")

flags.DEFINE_integer(
    "total_shards", default=1, help="total shards in overall processing"
)

flags.mark_flag_as_required("input_file")


def _tfrecord(record):
    uuid = record["uuid"]
    obj_uri = record["obj_uri"]
    sub_uri = record["sub_uri"]
    sub_surface = record["sub_label"]
    obj_surface = record["obj_label"]
    predicate_id = record["predicate_id"]
    # evidences_tf = []
    evidences = []
    for (i, sentence) in enumerate(record["templated_sentences"]):
        inputs_pretokenized = sentence.replace("<mask>", "<extra_id_0>")
        # for obj_surface in record['obj_labels']:
        targets_pretokenized = "<extra_id_0> " + obj_surface
        feature_dict = {
            "inputs_pretokenized": inputs_pretokenized,
            "targets_pretokenized": targets_pretokenized,
            "uuid": uuid,
            "obj_uri": obj_uri,
            "sub_uri": sub_uri,
            "predicate_id": predicate_id,
            "obj_surface": obj_surface,
            "sub_surface": sub_surface,
        }
        evidences.append(feature_dict)

    return evidences


def main(argv):
    data = []
    with open(FLAGS.input_file, "r") as f:
        # read data
        for line in f:
            examples = _tfrecord(json.loads(line))
            data.extend(examples)

        # write to json
        js_file = FLAGS.input_file.replace(".jsonl", ".jsonl.processed")
        with open(js_file, "w") as f:
            for example in data:
                json.dump(example, f)
                f.write("\n")

        # write to tf_record
        tf_file = FLAGS.input_file.replace("jsonl", "tfrecord")
        with tf.io.TFRecordWriter(tf_file) as writer:
            for example in data:
                obj = {
                    k: _bytes_feature(v.encode("utf-8"))
                    for (k, v) in example.items()
                }
                proto = tf.train.Example(
                    features=tf.train.Features(feature=obj)
                )
                writer.write(proto.SerializeToString())


if __name__ == "__main__":
    app.run(main)
