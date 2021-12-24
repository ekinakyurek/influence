import os
import json
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
    triples = abstract['triples']
    boundaries = abstract['sentences_boundaries']
    text = abstract['text']
    examples = []
    abstract_uri = abstract['uri'].split('/')[-1]
    for triple in triples:
        predicate = triple['predicate']
        predicate_code = predicate['uri'].split('/')[-1]
        
        object = triple['object']
        object_code = object['uri'].split('/')[-1]
        
        subject = triple['subject']
        subject_code = subject['uri'].split('/')[-1]
        
        if subject_code.startswith("XML") or object_code.startswith("XML"):
            continue
        
        sentence_id = triple['sentence_id']
        sentence_start, sentence_end = boundaries[sentence_id]
        sentence = text[sentence_start:sentence_end]
        
        fact = ",".join((predicate_code, object_code, subject_code))
        
        for i, target in enumerate((object, subject)):
            if target['boundaries'] is None:
                continue
            start, stop = target['boundaries']
            start = start - sentence_start
            stop = stop - sentence_start
            inputs_pretokenized = sentence[:start] + '<extra_id_0>' + sentence[stop:]
            targets_pretokenized = '<extra_id_0> ' + target['surfaceform']
            masked_uri = target['uri'].split('/')[-1]
            example = {'inputs_pretokenized': inputs_pretokenized,
                       'targets_pretokenized': targets_pretokenized,
                       'page_uri': abstract_uri,
                       'masked_uri': masked_uri,
                       'masked_type': "object" if i == 0 else "subject",
                       'fact': fact,
                       'sentence_uri': abstract_uri + f"-{sentence_id}-" + object_code + f"-{subject_code}-{i}",
                      }
            examples.append(example)

    return examples


def _tfrecord(record):
    feature = {k: _bytes_feature(v.encode('utf-8')) for (k, v) in record.items()}
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _filter_abstracts(abstracts, hashmap, query_file, used_facts):
    local_abstracts = []
    current_facts = set()
    print(f"processing: {query_file}")
    with open(query_file, 'r') as f:
        for line in f:
            question = json.loads(line)
            obj_uri = question['obj_uri']
            sub_uri = question['sub_uri']
            predicate_id = question['predicate_id']
            fact = ','.join((predicate_id, obj_uri, sub_uri))
            if fact not in current_facts:
                current_facts.add(fact)
                ids = hashmap.get(fact, [])
                local_abstracts.extend((abstracts[id] for id in ids))
                used_facts.add(fact)
    return local_abstracts


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
    abstracts = []
    with open(abstract_file, 'r') as f:
        for i, line in enumerate(f):
            abstracts.extend(json.loads(line))
    return abstracts


def dump_map_to_json(hashmap, output_file):
    with open(output_file, 'w') as f:
        json.dump(hashmap, f)


def _mask_all(abstracts):
    masked_abstracts = []
    for abstract in abstracts:
        masked_abstracts += _mask_an_abstract(abstract)
    return masked_abstracts


def process_a_file(abstracts,
                   hashmap,
                   fname,
                   used_facts):
    abstracts = _filter_abstracts(abstracts, hashmap, fname, used_facts)
    input_folder, fname = os.path.split(fname)
    output_folder = os.path.join(input_folder, 'abstracts/')
    os.makedirs(output_folder, exist_ok=True)
    output_json = os.path.join(output_folder, fname)
    dump_abstracts_json(abstracts, output_json)
    output_tf = os.path.join(output_folder, fname.replace('jsonl', 'tfrecord'))
    dump_abstracts_tf_record(abstracts, output_tf)
     
        
def unionize(abstracts):
    abstracts_map = {}
    for a in abstracts:
        identifier = a['inputs_pretokenized'] + "|||" + a['targets_pretokenized']
        if identifier not in abstracts_map:
            a['facts'] = a.pop('fact')
            a['sentence_uris'] = a.pop('sentence_uri')
            abstracts_map[identifier] = a
        else:
            abstracts_map[identifier]['facts'] += ";" + a.pop('fact')
            abstracts_map[identifier]['sentence_uris'] += ";" + a.pop('sentence_uri')
            
    return list(abstracts_map.values())


def main(argv):
    abstracts = read_all_abstracts(FLAGS.abstract_file)
    abstracts = unionize(_mask_all(abstracts))
    
    hashmap = {}
    for (i, abstract) in enumerate(abstracts):
        facts = abstract['facts'].split(';')
        for fact in facts:
            ids = hashmap.setdefault(fact, [])
            ids.append(i)
    
    fact_to_uri = {fact: [abstracts[id]['sentence_uris'] for id in ids] for fact, ids in hashmap.items()}
    
    dump_map_to_json(fact_to_uri,
                     os.path.join(FLAGS.input_folder,
                                  'abstracts',
                                  'hashmap.json'))
        
    used_facts = set([])
    
    for fname in listdir(FLAGS.input_folder):
        if fname.endswith('jsonl.processed') and 'abstracts' not in fname:
            file = os.path.join(FLAGS.input_folder, fname)
            process_a_file(abstracts, hashmap, file, used_facts)
        
    hashmap_used = {fact: hashmap.get(fact, []) for fact in used_facts}
    
    abstract_ids_used = set()
    
    for fact, ids in hashmap_used.items():
        for id in ids:
            abstract_ids_used.add(id)
            
    abstracts_used = [abstracts[id] for id in abstract_ids_used]
        
    fact_to_uri_used = {fact: [abstracts[id]['sentence_uris'] for id in ids] for fact, ids in hashmap_used.items()}
    
    dump_map_to_json(fact_to_uri_used,
                     os.path.join(FLAGS.input_folder,
                                  'abstracts',
                                  'hashmap_used.json'))

    
    output_json = os.path.join(FLAGS.input_folder,
                               'abstracts',
                               'all_used.jsonl')
    
    dump_abstracts_json(abstracts_used, output_json)
    
    output_tf = os.path.join(FLAGS.input_folder,
                             'abstracts',
                             'all_used.tfrecord')
    
    dump_abstracts_tf_record(abstracts_used, output_tf)
        
    output_json = os.path.join(FLAGS.input_folder, 'abstracts', 'all.jsonl')
    dump_abstracts_json(abstracts, output_json)
    output_tf = os.path.join(FLAGS.input_folder, 'abstracts', 'all.tfrecord')
    dump_abstracts_tf_record(abstracts, output_tf)
    

if __name__ == '__main__':
    app.run(main)
