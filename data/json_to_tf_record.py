from absl import app
from absl import flags
import json
from src.tf_utils import tf
from src.tf_utils import _bytes_feature

FLAGS = flags.FLAGS

flags.DEFINE_string('input_file', default=None,
                    help='input json file path to convert')

flags.mark_flag_as_required('input_file')


def main(argv):
    # write to json
    tf_file = FLAGS.input_file.replace('.jsonl', '.tfrecord')
    with tf.io.TFRecordWriter(tf_file) as writer:
        with open(FLAGS.input_file, 'r') as f:
            for line in f:
                example = json.loads(line)
                obj = {k: _bytes_feature(v.encode('utf-8'))
                       for (k, v) in example.items()}
                proto = tf.train.Example(features=tf.train.Features(feature=obj))
                writer.write(proto.SerializeToString())


if __name__ == '__main__':
    app.run(main)
