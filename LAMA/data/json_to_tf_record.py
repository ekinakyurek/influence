from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import json

FLAGS = flags.FLAGS

flags.DEFINE_string('input_file', default=None,
                    help='input json file path to convert')

flags.mark_flag_as_required('input_file')


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


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
