import argparse
from batch_eval_KB_completion import main as run_evaluation
from batch_eval_KB_completion import load_file
from batch_eval_KB_completion import lowercase_samples, filter_samples, parse_template
from lama.modules import build_model_by_name
from lama.utils import print_sentence_predictions, load_vocab
import lama.modules.base_connector as base
import pprint
import statistics
from os import listdir
import os, copy, sys
from os.path import isfile, join
from shutil import copyfile
from collections import defaultdict
import argparse, random, signal
import numpy as np
import torch
from my_utils import setLogger 
import logging, json

logger = logging.getLogger()

roberta_vocab = torch.load('pre-trained_language_models/roberta_vocab.save')

def get_TREx_parameters(data_path_pre="data/"):
    relations = load_file("{}relations.jsonl".format(data_path_pre))
    data_path_pre += "TREx/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post

def load_and_prepare_data(relations, data_path_pre, data_path_post):
    final_d = {}
    #vocab_subset = load_vocab(args.common_vocab_filename)
    
    #if args.fewshot_ft > 0 and args.fewshot_ft_mode == 'gather':
    #    fewshot_tr_gather, fewshot_dev_gather = [], []
        
    for r_count, relation in enumerate(relations):
        print('(load_and_prepare_data) relation idx:', r_count)
        #pp.pprint(relation)
        load_fn = "{}{}{}".format(data_path_pre, relation["relation"], data_path_post)
        template_cur = None
        if "template" in relation:
            template_cur = relation["template"]
        
        logger.info('loading %s', load_fn)
        # see if file exists
        try:
            data = load_file(load_fn)
        except Exception as e:
            print("Relation {} excluded because the file does not exist.".format(relation["relation"]))
            print("Exception: {}".format(e))
            continue
        
        #if args.lowercase:
        #    # lowercase all samples
        #    logger.info("lowercasing all samples...")
        #    all_samples = lowercase_samples(data, use_negated_probes=args.use_negated_probes)
        #else:
        #    # keep samples as they are
        all_samples = data
            
        #all_samples, ret_msg = filter_samples(model, data, vocab_subset, args.max_sentence_length, template_cur)
        if len(all_samples) < 100:
            logger.info('this relation is discarded because of its number < 100, num: %d, relation: %s', len(all_samples), str(relation))
            continue
        
        # if template is active (1) use a single example for (sub,obj) and (2) ...
        if template_cur is not None and template_cur != "":
            facts = []
            for sample in all_samples:
                sub = sample["sub_label"]
                obj = sample["obj_label"]
                if (sub, obj) not in facts:
                    facts.append((sub, obj))
            local_msg = "distinct template facts: {}".format(len(facts))
            logger.info("\n" + local_msg + "\n")
            print(local_msg)
            all_samples = []
            for f_i, fact in enumerate(facts):
                (sub, obj) = fact
                sample = {}
                sample["sub_label"] = sub
                sample["obj_label"] = obj
                # sobstitute all sentences with a standard template
                sample["masked_sentences"] = parse_template(
                    template_cur, sample["sub_label"].strip(), '<mask>' #base.MASK
                )
                sample['masked_sentence_ori'] = sample['masked_sentences'][0]
                if f_i == 0 or f_i % 200 == 0:
                    print('example', f_i, ':', sub, ',', obj, 'sentence:', sample['masked_sentences'])
                sample['relation'] = relation
                all_samples.append(sample)

        # create uuid if not present
        i = 0
        for sample in all_samples:
            if "uuid" not in sample:
                sample["uuid"] = i
            i += 1

        #deal with multi-target relations
        """
        samples_d = {}
        for sample in all_samples:
            assert(len(sample['masked_sentences']) == 1)
            ms = sample['masked_sentence_ori']
            if not ms in samples_d:
                sample['obj_labels'] = [sample['obj_label']]
                sample['obj_label'] = None 
                samples_d[ms] = sample
            else:
                samples_d[ms]['obj_labels'].append(sample['obj_label'])
        print('len of samples after merging', len(samples_d), 'original number of samples:', len(all_samples))
        for ms in samples_d:
            if len(samples_d[ms]['obj_labels']) > 1:
                print('debug: a sample of multiple-targets', samples_d[ms])
                break
        all_samples = list(samples_d.values())
        """
        for sample in all_samples:
            sample['obj_labels'] = [sample['obj_label']] #for convenience, here we don't deal with multi-target relations
 
        split_d = {} #split_d contains the final split results
        split_d['test_samples'] = all_samples
        #logger.info("\n" + ret_msg + "\n")
        final_d[relation['relation']] = split_d
    
    return final_d

exclude_list = [
        'P530', #'[X] maintains diplomatic relations with [Y] .'
        'P190', #'template': '[X] and [Y] are twin cities .'
        'P47', #'template': '[X] shares border with [Y] .'
        ]

def report_t4_oneword(t4_lis):
    res = []
    msg1 = 'report_t4_oneword: '
    msg2 = 'report_t4_invocab: '
    msg3 = 'report_t4_unique: '
    for k in range(4):
        dd = {'oneword_lis': [], 'oneword_unique_lis': [], 'invocab_lis': [], 'invocab_unique_lis': [], 'unique_lis': []}
        for t4 in t4_lis:
            ww = t4[k]
            if not ww in dd['unique_lis']: dd['unique_lis'].append(ww)
            if len(ww.split(' ')) != 1: continue
            dd['oneword_lis'].append(ww)
            if not ww in dd['oneword_unique_lis']: dd['oneword_unique_lis'].append(ww)
            if ww in roberta_vocab:
                dd['invocab_lis'].append(ww)
                if not ww in dd['invocab_unique_lis']: dd['invocab_unique_lis'].append(ww)

        #ll = [len(t4[k].split(' ')) == 1 for t4 in t4_lis]
        #num = sum(ll)
        msg1 += 'k{}: oneword_num {}  oneword_unique_num {}, '.format(k, len(dd['oneword_lis']), len(dd['oneword_unique_lis']))
        msg2 += 'k{}: invocab_num {}  invocab_unique_num {}, '.format(k, len(dd['invocab_lis']), len(dd['invocab_unique_lis']))
        msg3 += 'k{}: unique_num {}, '.format(k, len(dd['unique_lis']))

        res.append(dd)

    print(msg1)
    print(msg2)
    print(msg3)
    return res


def find_multi_hop(all_data):
    logger.info('Let us find multi-hop relations')
    obj_d, relation_d = {}, {}
    all_samples = []
    for r_n in all_data:
        if r_n in exclude_list:
            print('excluded!', r_n)
            continue
        samples = []
        for split in all_data[r_n]:
            samples.extend(all_data[r_n][split])
        relation_d[r_n] = samples[0]['relation']

        all_samples.extend(samples)
        for sample in samples:
            sub = sample['sub_label']
            for ii, obj in enumerate([sub] + sample['obj_labels']):
                if not obj in obj_d:
                    obj_d[obj] = []
                s_cur = copy.deepcopy(sample)
                if ii == 0: #mark whether it's sub or obj
                    s_cur['relation']['relation'] = 's' + s_cur['relation']['relation']
                else:
                    s_cur['relation']['relation'] = 'o' + s_cur['relation']['relation']
                obj_d[obj].append(s_cur)

    grandfather_d = {}

    for sample in all_samples:
        subobj = [sample['sub_label']] + sample['obj_labels']
        added = False
        for ii, so in enumerate(subobj):
            if so in obj_d:
                s_cur = copy.deepcopy(sample)
                if ii == 0: #mark whether it's sub or obj
                    s_cur['relation']['relation'] = 's' + s_cur['relation']['relation']
                else:
                    s_cur['relation']['relation'] = 'o' + s_cur['relation']['relation']
                for s_pre in obj_d[so]:
                    r_1, r_2 = s_pre['relation']['relation'], s_cur['relation']['relation']
                    #print(s_pre, '\n', '->', '\n', sample, '\n')
                    #if r_1 == r_2: continue
                    added = True
                    rr = r_1 + '_' + r_2
                    if rr not in grandfather_d:
                        grandfather_d[rr] = []
                    tt = [s_pre['sub_label'], s_pre['obj_labels'][0], s_cur['sub_label'], s_cur['obj_labels'][0]]
                    if len(set(tt)) <= 2: #they are just the same thing
                        continue
                    grandfather_d[rr].append((s_pre, s_cur))
                #if added: break
    
    if 'multihop' in MM:
        select_lis = {
            'P1376_P159':[], 
            'P37_P19':[],
            'P178_P108':[],
            'P527_P361':[],
            'P31_P527':[],
            'P31_P361':[],
            'P361_P361':[],
            'P527_P527':[],
            'P108_P108':[],
            }
    if '2i' in MM:
        select_lis = {
            ### next is cross probing
            'P19_P495': [],
            'P138_P937': [],
            'P176_P127': [],
            'P108_P127': [],
            'P101_P31': [],
            'P136_P31': [],
            'P138_P361': [],
            'P140_P361': [],
            'P31_P279': [],
            'P495_P495': [],
            'P108_P108': [],
            'P361_P361': [],
            'P131_P131': [],
            }
    
    # oP159_sP1376 k0 -> k3  The headquarter of [X] is in the country of [Y] .
    # P37_oP19 k2 -> k1  The official language of the country where [X] was born is [Y] .
    # oP178_oP108 k2 -> k0  [X] works for a company that developed [Y] . 
    # oP361_oP527 k0 -> k2 [X] is part of [Y] .
    # oP361_oP31 k2 -> k0 [Y] is part of [X] .
    # oP361_oP361 k0 -> k2 [X] and [Y] are part of the same thing .  #('Central Asia', 'Asia', 'South Asia', 'Asia') ('abstract algebra', 'mathematics', 'arithmetic', 'mathematics')
    # oP527_oP527 k0 -> k2 [X] and [Y] share at least one element . #('glycine', 'carbon', 'pectin', 'carbon') ('hydroxylamine', 'oxygen', 'carbonate', 'oxygen') 
    # oP178_oP178 k0 -> k2 The company that develops [X] also develops [Y] . #('macOS', 'Apple', 'MessagePad', 'Apple') ('Adobe Illustrator Artwork', 'Adobe', 'PostScript', 'Adobe')
    def in_select(rr, select_lis):
        r_1_n, r_2_n = rr.split('_')
        for ss in select_lis:
            #print(ss.split('_'))
            s0, s1 = ss.split('_')
            if (s0 in rr) and (s1 in rr) and (r_1_n[1:] in ss) and (r_2_n[1:] in ss):
                return ss
        return None
    
    def report_this(rr, print_num = 3):
        print(rr, len(grandfather_d[rr]))
        select_lis[ss].append((rr, len(grandfather_d[rr])))
        r_1_n, r_2_n = rr.split('_')
        print('r_1', relation_d[r_1_n[1:]])
        print('r_2', relation_d[r_2_n[1:]])
        t4_lis = []
        for s1, s2 in grandfather_d[rr]:
            t4 = (s1['sub_label'], s1['obj_labels'][0], s2['sub_label'], s2['obj_labels'][0])
            t4_lis.append(t4)
        for kk in range(print_num):
            print(t4_lis[kk])
        t4_res = report_t4_oneword(t4_lis)

    for rr in grandfather_d:
        if len(grandfather_d[rr]) < 50:
            continue
        ss = in_select(rr, select_lis)
        if ss is None:
            continue
        report_this(rr, print_num = 10)
        #if rr in ['P176_P108', 'P178_P108', 'P176_P127']:
        #    breakpoint()
        print()

    for ss in select_lis:
        print(ss, select_lis[ss])
    
    return grandfather_d, relation_d

multihop_lis = [
        {'rr': 'oP159_sP1376', 'idx': (0, 3), 'template': 'The headquarter of [X] is in the country of [Y] .'},
        {'rr': 'sP37_oP19', 'idx': (2, 1), 'template': 'The official language of the country where [X] was born is [Y] .'},
        {'rr': 'oP178_oP108', 'idx': (2, 0), 'template': '[X] works for a company that developed [Y] .'},
        {'rr': 'oP361_oP527', 'idx': (0, 2), 'template': '[X] is a low-level part of [Y] .'},
        {'rr': 'oP361_oP31', 'idx': (2, 0), 'template': 'One component of [X] is [Y] .'},
        {'rr': 'oP361_oP361', 'idx': (0, 2), 'template': '[X] and [Y] are part of the same thing .'},
        {'rr': 'oP527_oP527', 'idx': (0, 2), 'template': '[X] and [Y] share at least one element .'},
        {'rr': 'oP178_oP178', 'idx': (0, 2), 'template': '[X] and [Y] are developed by the same company .'},
    ]

multihop_onehop_lis = [
        {'rr': 'oP159_sP1376', 'idx': (2, 3), 'template': '[X] is the capital of [Y] .'},
        {'rr': 'sP37_oP19', 'idx': (0, 1), 'template': 'The official language of [X] is [Y] .'},
        {'rr': 'oP178_oP108', 'idx': (1, 0), 'template': '[Y] is developed by [X] .'},
        {'rr': 'oP361_oP527', 'idx': (3, 2), 'template': '[X] is a part of [Y] .'},
        {'rr': 'oP361_oP31', 'idx': (1, 0), 'template': '[Y] is part of [X] .'},
        {'rr': 'oP361_oP361', 'idx': (3, 2), 'template': '[Y] is part of [X] .'},
        {'rr': 'oP527_oP527', 'idx': (3, 2), 'template': '[Y] consists of [X] .'},
        {'rr': 'oP178_oP178', 'idx': (3, 2), 'template': '[Y] is developed by [X] .'},
    ]

trex_2i_lis = [
        {'rr': 'oP495_oP19', 'idx': (0, 1, 2), 'template': '[X] was created in [Y] , which is also the place [Z] was born in .'},
        {'rr': 'oP31_oP279', 'idx': (0, 1, 2), 'template': '[X] is a [Y] , of which [Z] is a subclass .'},
        {'rr': 'oP361_oP138', 'idx': (0, 1, 2), 'template': '[X] is part of [Y] , after which [Z] is named .'},
        {'rr': 'oP937_oP138', 'idx': (2, 3, 0), 'template': '[X] is named after [Y] , which is also the place [Z] used to work in .'},
        {'rr': 'oP31_oP101', 'idx': (0, 1, 2), 'template': '[X] is a [Y] , which is also the field [Z] works in .'},
        {'rr': 'oP127_oP176', 'idx': (0, 1, 2), 'template': '[X] is owned by [Y] , which produces [Z] .'},
        {'rr': 'oP127_oP108', 'idx': (0, 1, 2), 'template': '[X] is owned by [Y] , which [Z] works for .'},
        {'rr': 'oP361_oP140', 'idx': (0, 1, 2), 'template': '[X] is part of [Y] , which is the religion [Z] is affiliated with .'},
        {'rr': 'oP495_oP495', 'idx': (0, 1, 2), 'template': '[X] and [Z] were both created in [Y] .'},
        {'rr': 'oP131_oP131', 'idx': (0, 1, 2), 'template': '[X] and [Z] are both located in [Y] .'},
        {'rr': 'oP108_oP108', 'idx': (0, 1, 2), 'template': '[X] and [Z] both work for [Y] .'},
        #{'rr': 'sP361_sP361', 'idx': (1, 0, 3), 'template': '[Y] is part of [X] and [Z] .'}, #discarded because too few
    ]

def reformat_rr(rr_lis, rr_d, relation_d):
    res_rel_lis, res_rr_d = [], {}
    for item in rr_lis:
        rr = item['rr']
        sample_lis = []
        relation = {'relation': rr, 'template': item['template'], 'type': 'multihop', 'label': 'none', 'description': item['template']}
        print()
        print(relation)

        r_1_n, r_2_n = rr.split('_')
        print('r_1', relation_d[r_1_n[1:]])
        print('r_2', relation_d[r_2_n[1:]])
        unique_obj_lis = []
        uuid = 0
        
        tuple_d = {}
        for ii, ss in enumerate(rr_d[rr]):
            s0, s1 = ss
            idxs = item['idx']
            tt = [s0['sub_label'], s0['obj_labels'][0], s1['sub_label'], s1['obj_labels'][0]]
            sample = {'sub_label': tt[item['idx'][0]], 'obj_label': tt[item['idx'][1]], 'relation': relation, 'ss_from': ss}
            if len(idxs) == 3:
                sample['aux_label'] = tt[idxs[2]]
            ww = sample['obj_label']
            if len(ww.split(' ')) != 1: continue
            if not ww in roberta_vocab: continue
            if not ww in unique_obj_lis: unique_obj_lis.append(ww)
            
            tuple_id = sample['sub_label'] + '_' + sample['obj_label']
            if len(idxs) == 3:
                tuple_id += '_' + sample['aux_label']
            if tuple_id in tuple_d: continue
            tuple_d[tuple_id] = True
            
            if ii < 10:
                t0t1 = tt[idxs[0]]
                if len(idxs) == 3: t0t1 += ', ' + tt[idxs[2]]
                print(t0t1, '->', tt[item['idx'][1]], ' | original tuple:', str(tt), '| idx:', item['idx'])
            sample['uuid'] = rr + '_' + str(uuid)
            uuid = uuid + 1
            sample_lis.append(sample)
        
        print('len(sample_lis):', len(sample_lis), 'len(unique_obj_lis):', len(unique_obj_lis))
        print('unique_obj_lis[:50]:', unique_obj_lis[:50])
        
        res_rel_lis.append(relation)
        res_rr_d[rr] = sample_lis
    return res_rel_lis, res_rr_d

def dump_list_to_file(lis, out_fn):
    outf = open(out_fn, 'w')
    for item in lis:
        outf.write(json.dumps(item) + '\n')
    outf.close()

if __name__ == "__main__":
    #vocab_subset = load_vocab(args.common_vocab_filename)
    
    global MM
    #MM = 'trex_multihop'
    #MM = 'trex_multihop_onehop'
    MM = 'trex_2i'
    OUT_REL_FN = 'data/{}_relations.jsonl'.format(MM)
    OUT_DIR = 'data/{}/'.format(MM)
    DO_SAVE = True
    print('MM is', MM, 'DO_SAVE is', str(DO_SAVE))
    if input("Really start? (y/n)> ").lower().startswith('n'):
        sys.exit(1)
    os.system('mkdir -p ' + OUT_DIR)

    params = get_TREx_parameters()
    all_data_d = load_and_prepare_data(*params)
    rr_d, relation_d = find_multi_hop(all_data_d)

    if MM == 'trex_multihop': mm_lis = multihop_lis
    if MM == 'trex_multihop_onehop': mm_lis = multihop_onehop_lis
    if MM == 'trex_2i': mm_lis = trex_2i_lis

    res_rel_lis, res_rr_d = reformat_rr(mm_lis, rr_d, relation_d)
    
    if DO_SAVE == True:
        print('saving to', OUT_REL_FN)
        dump_list_to_file(res_rel_lis, OUT_REL_FN)
        for rr in res_rr_d:
            out_fn = OUT_DIR + rr + '.jsonl'
            print('saving to', out_fn)
            dump_list_to_file(res_rr_d[rr], out_fn)
