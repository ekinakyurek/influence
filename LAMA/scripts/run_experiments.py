import os
import sys
import random
import argparse
import json
import numpy as np
import torch


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--save_dir_root', type=str, default='./data/TREx_lama_templates_v3/')
parser.add_argument('--dataset', type=str, default='trex')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--relation_mode', type=str, default='template')
parser.add_argument('--template_source', type=str, default='lama')  # Lama or lpaqa or default (=>)

args = parser.parse_args()
args.command = ' '.join(sys.argv)


def parse_template(template, sample, subject_label, object_label):
    SUBJ_SYMBOL = "[X]"
    OBJ_SYMBOL = "[Y]"
    template = template.replace(SUBJ_SYMBOL, subject_label)
    template = template.replace(OBJ_SYMBOL, object_label)
    if '[Z]' in template:
        template = template.replace('[Z]', sample['aux_label'])
    return [template]


def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


def set_seed(args):
    print('setting all seed to', args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)


def load_vocab(vocab_filename):
    with open(vocab_filename, "r") as f:
        lines = f.readlines()
    vocab = [x.strip() for x in lines]
    return vocab


def load_and_prepare_data(relations, data_path_pre, data_path_post, args):
    final_d = {}

    for r_count, relation in enumerate(relations):
        print('(load_and_prepare_data) relation idx:', r_count)
  
        load_fn = "{}{}{}".format(data_path_pre, relation["relation"], data_path_post)

        print('loading %s', load_fn)
        # see if file exists
        try:
            all_samples = load_file(load_fn)
        except Exception as e:
            print("Relation {} excluded because the file does not exist.".format(relation["relation"]))
            print("Exception: {}".format(e))
            continue

        template_cur = None

        if "template" in relation:
            template_cur = relation["template"]

        # if template is active (1) use a single example for (sub,obj) and (2) ...
        if template_cur is not None and template_cur != "":
            facts = []
            added_samples = []
            for sample in all_samples:
                sub = sample["sub_label"]
                obj = sample["obj_label"]
                aux = sample['aux_label'] if 'aux_label' in sample else ''
                if (sub, obj, aux) not in facts:
                    facts.append((sub, obj, aux))
                    added_samples.append(sample)
                else:
                    print("===same relation exists===\n", sample)

            all_samples = []
            for f_i, fact in enumerate(facts):
                (sub, obj, aux) = fact
                sample = added_samples[f_i]
                sample["sub_label"] = sub
                sample["obj_label"] = obj
                if len(aux) > 0:
                    sample['aux_label'] = aux
                sample['template'] = template_cur
                sample["templated_sentences"] = parse_template(template_cur,
                                                               sample,
                                                               sample["sub_label"].strip(), 
                                                               "<mask>")
                if args.relation_mode == 'template':
                    sample["masked_sentences"] = parse_template(template_cur, sample, sample["sub_label"].strip(), "<mask>")
              
                sample['relation'] = relation
                all_samples.append(sample)

        for sample in all_samples:
            sample['masked_sentence_ori'] = sample['masked_sentences'][0]

        split_d = {}  # split_d contains the final split results
        random.shuffle(all_samples)
    
        samples_d = {}
        for sample in all_samples:
            assert(len(sample['masked_sentences']) == 1)
            ms = sample['masked_sentence_ori']
            if not ms in samples_d:
                sample['obj_labels'] = [sample['obj_label']]
                samples_d[ms] = sample
            else:
                samples_d[ms]['obj_labels'].append(sample['obj_label'])
      
        print('len of samples after merging', len(samples_d), 'original number of samples:', len(all_samples))
    
        for ms in samples_d:
            if len(samples_d[ms]['obj_labels']) > 1:
                print('debug: a sample of multiple-targets', samples_d[ms])
                break
          
        all_samples = list(samples_d.values())
        split_d['test_samples'] = all_samples
        print('!!! relation: %s num test_samples: %d', relation['relation'], len(all_samples))
        final_d[relation['relation']] = split_d
    
    for (k, v) in final_d.items():
        with open(os.path.join(args.save_dir_root, f'{k}.jsonl'), "w") as out_file:
            for example in final_d[k]['test_samples']:
                json.dump(example, out_file)
                out_file.write('\n')

    return final_d


def run_experiments(
    relations,
    data_path_pre,
    data_path_post,
    given_args=None,
):
    load_and_prepare_data(relations, data_path_pre, data_path_post, given_args)


def get_TREx_parameters(data_path_pre="data/"):
    relations = load_file("{}relations.jsonl".format(data_path_pre))
    data_path_pre += "TREx/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def run(parameters, d_n, given_args=None):
    args = given_args
    args.save_dir = args.save_dir_root
    os.system('mkdir -p ' + args.save_dir)
    set_seed(args)
    run_experiments(*parameters, given_args=args)
    print('re-print args', str(args))


if __name__ == "__main__":
    parameters = get_TREx_parameters()
    run(parameters, 'trex', given_args=args)
    print('%s', str(args))
