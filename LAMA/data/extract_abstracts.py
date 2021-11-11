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

flags.DEFINE_string('input_folder', default=None,
                    help='input file path to convert')

flags.mark_flag_as_required('input_folder')
flags.mark_flag_as_required('abstract_file')


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _get_abstract(record, abstracts, key='obj_uri'):
    uri = record[key]
    if uri in abstracts:
        abstract = abstracts[uri]
    else:
        abstract = next(iter(abstracts.values()))
    return abstract


def _mask_an_abstract(abstract):
    """Apply span masking to boundaries."""
    entities = abstract['entities']
    text = abstract['text']
    examples = []
    processed_boundaries = set()
    abstract_uri = abstract['uri'].split('/')[-1]
    for entity in entities:
        surfaceform = entity['surfaceform']
        start, stop = entity["boundaries"]
        if surfaceform is not None and (start, stop) not in processed_boundaries:
            inputs_pretokenized = text[:start] + '<extra_id_0>' + text[stop:]
            targets_pretokenized = '<extra_id_0> ' + surfaceform
            masked_uri = entity['uri'].split('/')[-1]
            example = {'inputs_pretokenized': inputs_pretokenized,
                       'targets_pretokenized': targets_pretokenized,
                       'page_uri': abstract_uri,
                       'masked_uri': masked_uri}
            processed_boundaries.add((start, stop))
            examples.append(example)

    return examples


def _tfrecord(record):
    feature = {k: _bytes_feature(v.encode('utf-8')) for (k, v) in record.items()}
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _filter_abstracts(abstracts_dict, query_file):
    abstracts = []
    print(f"processing: {query_file}")
    with open(query_file, 'r') as f:
        for line in f:
            question = json.loads(line)
            abstracts.append(_get_abstract(question,
                                           abstracts_dict,
                                           key='obj_uri'))
            abstracts.append(_get_abstract(question,
                                           abstracts_dict,
                                           key='sub_uri'))
    return abstracts


def dump_abstracts_json(abstracts, output_file):
    with open(output_file, 'w') as f:
        for abstract in abstracts:
            json.dump(abstract, f)
            f.write('\n')


def dump_abstracts_tf_record(abstracts, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for abstract in abstracts:
            tf_example = _tfrecord(abstract)
            writer.write(tf_example.SerializeToString())


def read_all_abstracts(abstract_file):
    abstracts_dict = {}
    with open(abstract_file, 'r') as f:
        for i, line in enumerate(f):
            abstracts = json.loads(line)
            for abstract in abstracts:
                if abstract['uri'] != abstract['docid']:
                    print(f'uri and docid is different uri: {abstract["uri"]}'
                          f' docid: {{abstract["uri"]}}')
                    uri = abstract['uri'].split('/')[-1]
                    if uri not in abstracts_dict:
                        abstracts_dict[uri] = abstract
                    else:
                        print(f'same uri in all.jsonl uri: {abstract["uri"]} no: {i}')
                        print(abstract['text'][:10])
    return abstracts_dict


def _mask_all(abstracts, fname, global_ids=None):
    masked_abstracts = []
    local_ids = set()
    for abstract in abstracts:
        uri = abstract['uri']
        if uri not in local_ids:
            masked_abstracts += _mask_an_abstract(abstract)
            local_ids.add(uri)
            if global_ids is not None:
                if uri not in global_ids:
                    global_ids.add(uri)
                else:
                    print(f'global id warning: uri: {uri} fname: {fname}')
        else:
            print(f'local id warning: uri: {uri} fname: {fname}')
    return masked_abstracts


def process_a_file(abstracts_dict,
                   fname,
                   global_ids=None):
    abstracts = _filter_abstracts(abstracts_dict, query_file=fname)
    masked_abstracts = _mask_all(abstracts, fname, global_ids)
    input_folder, fname = os.path.split(fname)
    output_folder = os.path.join(input_folder, 'abstracts/')
    os.makedirs(output_folder, exist_ok=True)
    output_json = os.path.join(output_folder, fname)
    dump_abstracts_json(masked_abstracts, output_json)
    output_tf = os.path.join(output_folder, fname.replace('jsonl', 'tfrecord'))
    dump_abstracts_tf_record(masked_abstracts, output_tf)


def main(argv):
    abstracts_dict = read_all_abstracts(FLAGS.abstract_file)
    global_ids = set()

    for fname in listdir(FLAGS.input_folder):
        if fname.endswith('jsonl') and 'abstracts' not in fname:
            file = os.path.join(FLAGS.input_folder, fname)
            process_a_file(abstracts_dict, file, global_ids)

    all_js = os.path.join(FLAGS.input_folder, 'abstracts', 'all.jsonl')
    all_tf = os.path.join(FLAGS.input_folder, 'abstracts', 'all.tfrecord')

    with open(all_js, 'w') as all_js_file:
        with tf.io.TFRecordWriter(all_tf) as all_tf_writer:
            with open(FLAGS.abstract_file, 'r') as f:
                for i, line in enumerate(f):
                    abstracts = json.loads(line)
                    print(i)
                    for abstract in abstracts:
                        for masked_abstract in _mask_an_abstract(abstract):
                            json.dump(masked_abstract, all_js_file)
                            all_js_file.write('\n')
                            tf_example = _tfrecord(masked_abstract)
                            all_tf_writer.write(tf_example.SerializeToString())


def main_v2(argv):
    uuids = set()
    abstracts = []
    for fname in listdir(os.path.join(FLAGS.input_folder, 'abstracts/')):
        if fname.endswith('jsonl') and fname.startswith('P'):
            file = os.path.join(FLAGS.input_folder, 'abstracts', fname)
            print(file)
            with open(file, 'r') as f:
                for line in f:
                    abstract = json.loads(line)
                    uuid = (abstract["inputs_pretokenized"],
                            abstract["targets_pretokenized"],
                            abstract["page_uri"],
                            abstract["masked_uri"])
                    if uuid not in uuids:
                        abstracts.append(abstract)
                        uuids.add(uuid)

    all_js = os.path.join(FLAGS.input_folder, 'abstracts', 'all_used.jsonl')
    all_tf = os.path.join(FLAGS.input_folder, 'abstracts', 'all_used.tfrecord')
    with open(all_js, 'w') as all_js_file:
        with tf.io.TFRecordWriter(all_tf) as all_tf_writer:
            for abstract in abstracts:
                json.dump(abstract, all_js_file)
                all_js_file.write('\n')
                tf_example = _tfrecord(abstract)
                all_tf_writer.write(tf_example.SerializeToString())


if __name__ == '__main__':
    # app.run(main)
    app.run(main_v2)
