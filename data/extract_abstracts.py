import json
import os
from os import listdir
from absl import app, flags
from tqdm import tqdm
from src.json_utils import dump_abstracts_json, dump_map_to_json
from src.lama_utils import get_sentence


FLAGS = flags.FLAGS


def _mask_an_abstract(abstract):
    """Apply span masking to boundaries."""
    triples = abstract["triples"]
    boundaries = abstract["sentences_boundaries"]
    text = abstract["text"]
    examples = []
    abstract_uri = abstract["uri"].split("/")[-1]
    for triple in triples:
        predicate = triple["predicate"]
        predicate_code = predicate["uri"].split("/")[-1]

        object = triple["object"]
        object_code = object["uri"].split("/")[-1]

        subject = triple["subject"]
        subject_code = subject["uri"].split("/")[-1]

        if subject_code.startswith("XML") or object_code.startswith("XML"):
            continue

        sentence_id = triple["sentence_id"]
        sentence_start, sentence_end = boundaries[sentence_id]
        sentence = text[sentence_start:sentence_end]

        fact = ",".join((predicate_code, object_code, subject_code))

        for i, target in enumerate((object, subject)):
            if target["boundaries"] is None:
                continue
            start, stop = target["boundaries"]
            start = start - sentence_start
            stop = stop - sentence_start
            inputs_pretokenized = (
                sentence[:start] + "<extra_id_0>" + sentence[stop:]
            )
            targets_pretokenized = "<extra_id_0> " + target["surfaceform"]
            masked_uri = target["uri"].split("/")[-1]
            example = {
                "inputs_pretokenized": inputs_pretokenized,
                "targets_pretokenized": targets_pretokenized,
                "page_uri": abstract_uri,
                "masked_uri": masked_uri,
                "masked_type": "object" if i == 0 else "subject",
                "fact": fact,
                "example_uri": abstract_uri
                + f"-{sentence_id}-"
                + object_code
                + f"-{subject_code}-{i}",
            }
            examples.append(example)

    return examples


def _filter_abstracts(abstracts, fact_to_ids, query_file, used_facts):
    local_abstracts = []
    current_facts = set()
    print(f"processing: {query_file}")
    with open(query_file, "r") as f:
        for line in f:
            question = json.loads(line)
            obj_uri = question["obj_uri"]
            sub_uri = question["sub_uri"]
            predicate_id = question["predicate_id"]
            fact = ",".join((predicate_id, obj_uri, sub_uri))
            if fact not in current_facts:
                current_facts.add(fact)
                ids = fact_to_ids.get(fact, [])
                local_abstracts += [abstracts[id] for id in ids]
                used_facts.add(fact)
    return local_abstracts


def read_all_abstracts(abstract_file):
    abstracts = []
    with open(abstract_file, "r") as f:
        for i, line in enumerate(f):
            print(i)
            abstracts += json.loads(line)
    return abstracts


def _mask_all(abstracts):
    masked_abstracts = []
    for abstract in abstracts:
        masked_abstracts += _mask_an_abstract(abstract)
    return masked_abstracts


def process_a_file(abstracts, fact_to_ids, fname, used_facts):
    abstracts = _filter_abstracts(abstracts, fact_to_ids, fname, used_facts)
    input_folder, fname = os.path.split(fname)
    output_folder = os.path.join(input_folder, "abstracts/")
    os.makedirs(output_folder, exist_ok=True)
    output_json = os.path.join(output_folder, fname)
    dump_abstracts_json(abstracts, output_json)
    # output_tf = os.path.join(output_folder, fname.replace("jsonl", "tfrecord"))
    # dump_abstracts_tf_record(abstracts, output_tf)


def unionize(abstracts):
    abstracts_map = {}
    sentence_to_facts = {}

    for a in tqdm(abstracts):
        identifier = (
            a["inputs_pretokenized"] + "|||" + a["targets_pretokenized"]
        )

        if identifier not in abstracts_map:
            a["example_uris"] = a.pop("example_uri")
            abstracts_map[identifier] = a
        else:
            new_uri = a.pop("example_uri")
            uris = abstracts_map[identifier]["example_uris"]
            if new_uri not in uris:
                uris += ";" + new_uri
                abstracts_map[identifier]["example_uris"] = uris

        sentence_identifier = get_sentence(a)
        if sentence_identifier not in sentence_to_facts:
            sentence_to_facts[sentence_identifier] = a.pop("fact")
        else:
            new_fact = a.pop("fact")
            facts = sentence_to_facts[sentence_identifier]

            if new_fact not in facts:
                facts += ";" + new_fact
                sentence_to_facts[sentence_identifier] = facts

        abstracts_map[identifier]["facts"] = sentence_to_facts[
            sentence_identifier
        ]

    abstracts = list(abstracts_map.values())

    return abstracts


def get_fact_to_ids_map(abstracts):
    fact_to_ids = {}
    for (i, abstract) in enumerate(abstracts):
        facts = abstract["facts"].split(";")
        abstract["id"] = i  # ids based on indices
        for fact in facts:
            ids = fact_to_ids.setdefault(fact, [])
            ids.append(i)
    return fact_to_ids


def main(argv):
    abstracts = read_all_abstracts(FLAGS.abstract_file)
    abstracts = unionize(_mask_all(abstracts))

    fact_to_ids = get_fact_to_ids_map(abstracts)

    dump_map_to_json(
        fact_to_ids,
        os.path.join(FLAGS.input_folder, "abstracts", "fact_to_ids.json"),
    )

    used_facts = set([])

    for fname in listdir(FLAGS.input_folder):
        if fname.endswith("jsonl.processed") and "abstracts" not in fname:
            file = os.path.join(FLAGS.input_folder, fname)
            process_a_file(abstracts, fact_to_ids, file, used_facts)

    fact_to_ids_used = {fact: fact_to_ids.get(fact, []) for fact in used_facts}

    abstract_ids_used = set()

    for fact, ids in fact_to_ids_used.items():
        for id in ids:
            abstract_ids_used.add(id)

    abstracts_used = [abstracts[id] for id in abstract_ids_used]

    # fact_to_uris_used = {
    #     fact: [abstracts[id]["example_uris"] for id in ids]
    #     for fact, ids in fact_to_ids_used.items()
    # }

    dump_map_to_json(
        fact_to_ids_used,
        os.path.join(FLAGS.input_folder, "abstracts", "fact_to_ids_used.json"),
    )

    output_json = os.path.join(
        FLAGS.input_folder, "abstracts", "all_used_v2.jsonl"
    )

    dump_abstracts_json(abstracts_used, output_json)

    # output_tf = os.path.join(
    #     FLAGS.input_folder, "abstracts", "all_used.tfrecord"
    # )

    # dump_abstracts_tf_record(abstracts_used, output_tf)

    output_json = os.path.join(FLAGS.input_folder, "abstracts", "all_v2.jsonl")
    dump_abstracts_json(abstracts, output_json)

    # output_tf = os.path.join(FLAGS.input_folder, "abstracts", "all.tfrecord")
    # dump_abstracts_tf_record(abstracts, output_tf)


if __name__ == "__main__":

    flags.DEFINE_string(
        "abstract_file", default=None, help="input file path to convert"
    )

    flags.DEFINE_string(
        "input_folder", default=None, help="input file path to convert"
    )

    flags.mark_flag_as_required("input_folder")

    flags.mark_flag_as_required("abstract_file")
    app.run(main)
