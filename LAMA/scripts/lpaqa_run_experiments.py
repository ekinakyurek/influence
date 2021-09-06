# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
from batch_eval_KB_completion import main as run_evaluation
from batch_eval_KB_completion import load_file
from batch_eval_KB_completion import lowercase_samples, filter_samples, parse_template
from lama.modules import build_model_by_name
from lama.utils import print_sentence_predictions, load_vocab, str2bool
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
import logging
#import extract_multi_hop

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger()

def set_seed(args):
    print('setting all seed to', args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed) 

def exit_gracefully(signum, frame):
    # restore the original signal handler as otherwise evil things will happen
    # in raw_input when CTRL+C is pressed, and our signal handler is not re-entrant
    signal.signal(signal.SIGINT, original_sigint)

    try:
        if input("Catched CTRL-C\nyour command is {} \nReally quit? (y/n)> ".format(' '.join(sys.argv))).lower().startswith('y'):
            sys.exit(1)

    except KeyboardInterrupt:
        print("Ok ok, quitting")
        sys.exit(1)

    # restore the exit gracefully handler here    
    signal.signal(signal.SIGINT, exit_gracefully)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--save_dir_root', type=str, default='./experiments/base/')
parser.add_argument('--dataset', type=str, default='trex')
parser.add_argument('--fewshot_context', type=int, default = -1)
parser.add_argument('--threads', type=int, default = 10)
parser.add_argument('--seed', type = int, default=42)
parser.add_argument('--max_sentence_length', type = int, default = 100, help='used for filtering samples')
parser.add_argument("--common_vocab_filename", type = str, default = "pre-trained_language_models/common_vocab_cased.txt")
parser.add_argument('--use_common_vocab', type = str2bool, default = False)
parser.add_argument('--only_run_num', type = int, default = -1) #for debug or tuning, i only run for a few relations
parser.add_argument('--relation_mode', type = str, default = 'template') #template, or relvec
parser.add_argument('--relvec_position', type = str, default = 'M') #F(front)M(middle)E(end)
parser.add_argument('--relvec_max_num', type = int, default = 60)
parser.add_argument('--relvec_num', type = int, default = 5)
parser.add_argument('--relvec_initialize_scale', type = float, default = 1) #1 >(a little better than) 0.1 > 0.01
parser.add_argument('--relvec_initialize_mode', type = str, default = 'mean_embed') #gaussian, or mean_embed, or from_template
parser.add_argument('--relvec_input_mode', type = str, default = 'direct') #direct or mlp
parser.add_argument('--relvec_mlp_final_tanh', type = str2bool, default = True)
parser.add_argument('--relvec_mlp_hidden_dim', type = int, default = 512)
parser.add_argument('--relvec_mlp_layer_num', type = int, default = 1) 
parser.add_argument('--fewshot_ft', type=int, default = -1) #fewshot finetune
parser.add_argument('--fewshot_ft_mode', type=str, default='each') #each (finetune for each relation) or gather (first gather examples from all relations, and just do training one time)
parser.add_argument('--fewshot_ft_batch_mode', type = str, default = 'full_batch') #random or full_batch
parser.add_argument('--fewshot_ft_param_mode', type=str, default='all') #could be all or only_final_bias or fix_final_bias or fix_lm_head or only_relvec or first_relvec_then_all
parser.add_argument('--fewshot_ft_param_mode_cur', type=str, default=None) #Just a placeholder, don't use it directly
parser.add_argument('--fewshot_ft_phase1_data_num', type=int, default=-1) #for phase1, just use this number of data for finetuning
parser.add_argument('--fewshot_ft_param_all_lr_overwrite', type = str2bool, default=False)
parser.add_argument('--fewshot_ft_bz', type = int, default = 8)
parser.add_argument('--fewshot_ft_devnum', type = int, default = 10)
parser.add_argument('--fewshot_ft_maxsteps', type = int, default = 1000)
parser.add_argument('--fewshot_ft_tuneseed_num', type = int, default = 1) #will try different seed, and give the best result
parser.add_argument('--fewshot_ft_shuffledata_seednum', type = int, default = 1)
parser.add_argument('--fewshot_ft_tune_mode', type = str, default = 'justseed') #just seed, or shuffledata

parser.add_argument('--lpaqa_tune_num', type = int, default = 0)
#the following are all used for fewshot_ft

parser.add_argument("--dry_run", action="store_true", default = False)
parser.add_argument('--weight_decay', type = float, default = 0.1)
parser.add_argument('--learning_rate', type = float, default = 1e-06)
parser.add_argument('--relvec_learning_rate', type = float, default = 0.0001)
parser.add_argument('--relvec_weight_decay', type = float, default = 0.01)
parser.add_argument('--warmup_steps', type = int, default = 10)
parser.add_argument('--max_training_steps', type = int, default = 1000)
parser.add_argument('--adam_epsilon', type = float, default = 1e-08)
parser.add_argument('--max_grad_norm', type = float, default = 1.0)

#parser.add_argument("--play_find_multi_hop", action="store_true", default = False)

args = parser.parse_args()

if args.fewshot_ft_devnum > args.fewshot_ft and args.fewshot_ft > 0:
    logger.info('===WARNING=== setting ft_devnum from %d to %d', args.fewshot_ft_devnum, args.fewshot_ft)
    args.fewshot_ft_devnum = args.fewshot_ft
if args.fewshot_ft_tuneseed_num != 1:
    pass
    #assert(args.fewshot_ft_param_mode == 'only_relvec')

if args.use_common_vocab == False:
    logger.info('\n===WARNING=== not using common_vocab!!!\n')
    args.common_vocab_filename = 'none'

args.command = ' '.join(sys.argv)
print('args', args)

if args.relation_mode == 'relvec':
    assert(args.fewshot_ft_param_mode in ['only_relvec', 'first_relvec_then_all', 'all'])
os.system('mkdir -p ' + args.save_dir_root)

LMs = [
    #{
    #    "lm": "bert",
    #    "label": "bert_large",
    #    "models_names": ["bert"],
    #    "bert_model_name": "bert-large-cased",
    #    "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
    #},
    {
        "lm": "roberta",
        "label": "roberta_large",
        "models_names": ["roberta"],
        "roberta_model_name": "roberta-large",
    },
]
"""
{
    "lm": "bert",
    "label": "bert_base",
    "models_names": ["bert"],
    "bert_model_name": "bert-base-cased",
    "bert_model_dir": "pre-trained_language_models/bert/cased_L-12_H-768_A-12",
},
"""

def normweighted_sum(p_lis, w_lis):
    p = torch.FloatTensor(p_lis)
    w = torch.FloatTensor(w_lis)
    w = w / torch.sum(w).item()
    assert(len(p_lis) == len(w_lis))
    return torch.sum(p * w).item()

pp = pprint.PrettyPrinter(width=80, compact=True)

def load_and_prepare_data(relations, data_path_pre, data_path_post, model, args):
    final_d = {}
    vocab_subset = load_vocab(args.common_vocab_filename) if args.use_common_vocab else None
    
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
        
        lpaqa_fn_1 = 'lpaqa_prompts/mine/{}.jsonl'.format(relation['relation'])
        lpaqa_fn_2 = 'lpaqa_prompts/paraphrase/{}.jsonl'.format(relation['relation'])
        try:
            lpaqa_templates = []
            for lpaqa_fn in [lpaqa_fn_1, lpaqa_fn_2]:
                print('loading lpaqa templates from', lpaqa_fn)
                lpaqa_templates.extend(load_file(lpaqa_fn))
        except Exception as e:
            print('Exception! Something Wrong?')
            breakpoint()
        assert(template_cur is not None)
        temp_lis = [template_cur] + [a['template'] for a in lpaqa_templates]

        #if args.lowercase:
        #    # lowercase all samples
        #    logger.info("lowercasing all samples...")
        #    all_samples = lowercase_samples(data, use_negated_probes=args.use_negated_probes)
        #else:
        #    # keep samples as they are
        all_samples = data
            
        all_samples, ret_msg = filter_samples(model, data, vocab_subset, args.max_sentence_length, template_cur)
        if len(all_samples) < 50:
            logger.info('this relation is discarded because of its number < 50, num: %d, relation: %s', len(all_samples), str(relation))
            breakpoint()
            continue
        
        all_samples_ori = all_samples
        random.shuffle(all_samples_ori)

        # if template is active (1) use a single example for (sub,obj) and (2) ...
        #if template_cur is not None and template_cur != "":
        temp_data_d = {}
        for template_cur in temp_lis:
            print(relation['relation'], 'current template:', template_cur)
            all_samples = copy.deepcopy(all_samples_ori)
            facts = []
            for sample in all_samples:
                sub = sample["sub_label"]
                obj = sample["obj_label"]
                aux = sample['aux_label'] if 'aux_label' in sample else ''
                if (sub, obj, aux) not in facts:
                    facts.append((sub, obj, aux))
            local_msg = "distinct template facts: {}".format(len(facts))
            logger.info("\n" + local_msg + "\n")
            print(local_msg)
            all_samples, relvec_last_co = [], -1
            for f_i, fact in enumerate(facts):
                (sub, obj, aux) = fact
                sample = {}
                sample["sub_label"] = sub
                sample["obj_label"] = obj
                if len(aux) > 0: sample['aux_label'] = aux
                sample['template'] = template_cur
                sample["templated_sentences"] = parse_template(template_cur, sample, sample["sub_label"].strip(), model.mask_token)
                if args.relation_mode == 'template':
                    # sobstitute all sentences with a standard template
                    sample["masked_sentences"] = parse_template(template_cur, sample, sample["sub_label"].strip(), model.mask_token) #base.MASK

                if args.relation_mode == 'relvec':
                    if args.relvec_initialize_mode == 'from_template':
                        #we will convert to relvec in the roberta_connector
                        sample['masked_sentences'] = sample["templated_sentences"] 
                    else:
                        ss, relvec_idx = '', 0
                        if 'F' in args.relvec_position:
                            for k in range(args.relvec_num):
                                ss = ss + model.relvec_tokens[relvec_idx] + ' '
                                relvec_idx += 1
                        
                        ss += sub + ' '
                        if 'M' in args.relvec_position:
                            for k in range(args.relvec_num):
                                ss = ss + model.relvec_tokens[relvec_idx] + ' '
                                relvec_idx += 1
                        
                        if len(aux) > 0:
                            ss += aux + ' '
                            for k in range(args.relvec_num):
                                ss = ss + model.relvec_tokens[relvec_idx] + ' '
                                relvec_idx += 1

                        ss += model.mask_token 
                        if 'E' in args.relvec_position:
                            for k in range(args.relvec_num):
                                ss = ss + ' ' + model.relvec_tokens[relvec_idx]
                                relvec_idx += 1
                        sample['masked_sentences'] = [ss]
                        tokenized_ss = model.tokenizer(ss)['input_ids'] #check that the vec is there after tokenization
                        assert(model.mask_token_id in tokenized_ss and model.relvec_idxs[0] in tokenized_ss)

                if (f_i == 0 or f_i % 200 == 0) and f_i < 2000:
                    print('example', f_i, ':', sub, ',', obj, 'sentence:', sample['masked_sentences'])
                sample['relation'] = relation
                all_samples.append(sample)

            # create uuid if not present
            i = 0
            for sample in all_samples:
                if "uuid" not in sample:
                    sample["uuid"] = i
                i += 1

            for sample in all_samples:
                sample['masked_sentence_ori'] = sample['masked_sentences'][0]
            split_d = {} #split_d contains the final split results
     
            fewshot_ft_samples, fewshot_objs_d, fewshot_context_s = [], {}, ''
            #random.shuffle(all_samples) #I only do this once in the beginning
            if args.fewshot_context > 0:
                logger.info('processing fewshot_context %d ...', args.fewshot_context)
                for k in range(args.fewshot_context):
                    sample = all_samples[k]
                    assert(len(sample['masked_sentences']) == 1)
                    ms = sample['masked_sentences'][0]
                    assert(('[MASK]' in ms) or ('<mask>' in ms))
                    fewshot_context_s += ms.replace('[MASK]', sample['obj_label']).replace('<mask>', sample['obj_label']) + ' '
                    if ms not in fewshot_objs_d: fewshot_objs_d[ms] = []
                    fewshot_objs_d[ms].append(sample['obj_label'])
                #context_s = context_s.replace('.', ',') 
                print('removing fewshot_context examples from the all_samples (they will not appear in multi-target samples)...')
                all_samples = all_samples[args.fewshot_context:]
                logger.info('prefixing all samples with: %s', fewshot_context_s)
                for sample in all_samples:
                    assert(len(sample['masked_sentences']) == 1)
                    sample['masked_sentences'][0] = fewshot_context_s + sample['masked_sentences'][0]
                
            if args.fewshot_ft > 0:
                assert(args.fewshot_ft < len(all_samples))
                for k in range(args.fewshot_ft):
                    sample = all_samples[k]
                    fewshot_ft_samples.append(sample)
                    assert(len(sample['masked_sentences']) == 1)
                    ms = sample['masked_sentences'][0]
                    if ms not in fewshot_objs_d: fewshot_objs_d[ms] = []
                    fewshot_objs_d[ms].append(sample['obj_label'])
                #context_s = context_s.replace('.', ',') 
                print('removing fewshot examples from the test-set (they will not appear in multi-target samples)...')
                all_samples = all_samples[args.fewshot_ft:]
                assert(args.fewshot_ft_devnum < len(all_samples))     
                
                fewshot_ft_samples_dev = copy.deepcopy(all_samples[:args.fewshot_ft_devnum])
                
                split_d['fewshot_samples'] = fewshot_ft_samples
                split_d['fewshot_samples_dev'] = fewshot_ft_samples_dev
                #if args.fewshot_ft_mode == 'gather':
                #    fewshot_tr_gather.extend(fewshot_samples)
                #    fewshot_dev_gather.extend(fewshot_samples_dev)
     
            #deal with multi-target relations
            samples_d = {}
            for sample in all_samples:
                assert(len(sample['masked_sentences']) == 1)
                ms = sample['masked_sentence_ori']
                if not ms in samples_d:
                    sample['obj_labels'] = [sample['obj_label']]
                    if fewshot_objs_d is not None and ms in fewshot_objs_d:
                        sample['fewshot_obj_labels'] = fewshot_objs_d[ms]
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
            
            if args.dataset == 'trex_2i' and len(all_samples) > 1000:
                print('shrinking all_samples to 1k...')
                all_samples = all_samples[:1000]

            split_d['test_samples'] = all_samples
            #logger.info("\n" + ret_msg + "\n")
            temp_data_d[template_cur] = split_d
        
        final_d[relation['relation']] = temp_data_d
    
    breakpoint() #save this final d!
    #torch.save(final_d, 'lpaqa_prompts/loaded_templates.save')
    return final_d

def run_experiments(
    relations,
    data_path_pre,
    data_path_post,
    input_param={
        "lm": "bert",
        "label": "bert_large",
        "models_names": ["bert"],
        "bert_model_name": "bert-large-cased",
        "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
    },
    use_negated_probes=False,
    given_args = None,
):
    input_param.update(given_args.__dict__)
    model_args = argparse.Namespace(**input_param)
    [model_type_name] = model_args.models_names
    model = build_model_by_name(model_type_name, model_args)
    model.try_cuda()

    all_Precision1, all_MRR, all_count = [], [], []
    type_Precision1 = defaultdict(list)
    type_count = defaultdict(list)

    results_file = open("last_results.csv", "w+")
    
    #all_data_d = load_and_prepare_data(relations, data_path_pre, data_path_post, model, given_args)
    load_fn = 'lpaqa_prompts/loaded_templates.save'
    all_data_d = torch.load(load_fn)
    
    #if given_args.play_find_multi_hop:
    #    play_multi_hop.find_multi_hop(all_data_d) 
    #    sys.exit(0)

    if given_args.fewshot_ft > 0 and given_args.fewshot_ft_mode == 'gather':
        fewshot_tr_gather, fewshot_dev_gather = [], []
        for r_idx, relation in enumerate(relations):
            if relation['relation'] in all_data_d:
                data_d = all_data_d[relation['relation']]
                fewshot_tr_gather.extend(data_d['fewshot_samples'])
                fewshot_dev_gather.extend(data_d['fewshot_samples_dev'])
        logger.info('len(fewshot_tr_gather): %d len(fewshot_dev_gather): %d', len(fewshot_tr_gather), len(fewshot_dev_gather))
        logger.info('fewshot_ft gather mode, starting fine-tuning model...')
        fewshot_ft_res = model.mlm_finetune(fewshot_tr_gather, fewshot_dev_gather, given_args)  

    all_res_d, best_temp_d = {}, {}
    for r_count, relation in enumerate(relations):
        logger.info('relation idx: %d', r_count)
        if given_args.only_run_num > 0 and r_count >= given_args.only_run_num:
            break
        pp.pprint(relation)
        PARAMETERS = {
            "dataset_filename": "{}{}{}".format(
                data_path_pre, relation["relation"], data_path_post
            ),
            "common_vocab_filename": "pre-trained_language_models/common_vocab_cased.txt",
            "template": "",
            "bert_vocab_name": "vocab.txt",
            "batch_size": 32,
            "logdir": "output",
            "full_logdir": "output/results/{}/{}".format( #full_logdir specified here!
                input_param["label"], relation["relation"]
            ),
            "lowercase": False,
            "max_sentence_length": 100,
            "threads": -1,
            "interactive": False,
            "use_negated_probes": use_negated_probes,
        }

        if "template" in relation:
            PARAMETERS["template"] = relation["template"]
            if use_negated_probes:
                PARAMETERS["template_negated"] = relation["template_negated"]
        
        PARAMETERS.update(input_param)
        if given_args is not None:
            PARAMETERS.update(given_args.__dict__)
        print(PARAMETERS)

        args = argparse.Namespace(**PARAMETERS)

        """
        print('loading', args.dataset_filename)
        # see if file exists
        try:
            data = load_file(args.dataset_filename)
        except Exception as e:
            print("Relation {} excluded.".format(relation["relation"]))
            print("Exception: {}".format(e))
            continue
        
        if args.lowercase:
            # lowercase all samples
            logger.info("lowercasing all samples...")
            all_samples = lowercase_samples(data, use_negated_probes=args.use_negated_probes)
        else:
            # keep samples as they are
            all_samples = data
        
        vocab_subset = None
        vocab_subset = load_vocab(args.common_vocab_filename)
        all_samples, ret_msg = filter_samples(model, data, vocab_subset, args.max_sentence_length, args.template)
        if len(all_samples) < 100:
            logger.info('this relation is discarded because of its number < 100, num: %d, relation: %s', len(all_samples), str(relation))
            continue
        #logger.info("\n" + ret_msg + "\n")
        """
        if not relation['relation'] in all_data_d: continue
        
        t_d =  all_data_d[relation['relation']]
        best_mrr, best_p, best_results, lis_p, best_data_d, best_template = -1, -1, None, [], None, None
        for template_cur in t_d:
            data_d_cur = t_d[template_cur]
            assert(len(data_d_cur['test_samples']) > args.lpaqa_tune_num)
            data_d_cur_clone = copy.deepcopy(data_d_cur)
            print('truncating test_samples to', args.lpaqa_tune_num)
            data_d_cur['test_samples'] = data_d_cur['test_samples'][:args.lpaqa_tune_num]
            Precision1, all_results = run_evaluation(args, shuffle_data = False, model = model, data_d = data_d_cur)
            mrr = all_results['global_MRR']
            lis_p.append(Precision1)
            if mrr > best_mrr:
                best_mrr, best_p, best_template = mrr, Precision1, template_cur
                best_results, best_data_d = all_results, data_d_cur_clone
        
        #Precision1, all_results = best_p, best_results
        print('list of precisions for different templates:', lis_p)
        print('reevaluating with best_data_d!')
        Precision1, all_results = run_evaluation(args, shuffle_data = False, model = model, data_d = best_data_d)
        best_temp_d[relation['relation']] = best_template

        num_cur = len(all_results['list_of_results'])
        MRR = all_results['global_MRR']
        print("P@1 : {} MRR: {}".format(Precision1, MRR), flush=True)
        all_Precision1.append(Precision1)
        all_MRR.append(MRR)
        all_count.append(num_cur)
        
        all_results['relation'] = relation
        all_res_d[relation['relation']] = all_results
        
        results_file.write(
            "{},{}\n".format(relation["relation"], round(Precision1 * 100, 2))
        )
        results_file.flush()

        if "type" in relation:
            type_Precision1[relation["type"]].append(Precision1)
            data = load_file(PARAMETERS["dataset_filename"])
            type_count[relation["type"]].append(num_cur) #len(data))
    
    #end of "for relation in relations: "
    print()
    print('count list for each relation:', all_count)
    #mean_p1, mean_mrr = normweighted_sum(all_Precision1, all_count), normweighted_sum(all_MRR, all_count)  #statistics.mean(all_Precision1)
    mean_p1, mean_mrr = np.mean(all_Precision1), np.mean(all_MRR)
    print("@@@ {} - mean P@1: {} mean MRR: {}".format(input_param["label"], mean_p1, mean_mrr))
    results_file.close()

    for t, l in type_Precision1.items():

        print(
            "@@@ ",
            input_param["label"],
            t,
            np.mean(l),
            #normweighted_sum(l, type_count[t]), #statistics.mean(l), 
            sum(type_count[t]),
            len(type_count[t]),
            flush=True,
        )

    return all_res_d, mean_p1, mean_mrr, all_Precision1, type_Precision1, type_count, best_temp_d


def get_TREx_parameters(data_path_pre="data/"):
    relations = load_file("{}relations.jsonl".format(data_path_pre))
    data_path_pre += "TREx/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post

def get_TREx_multihop_parameters(data_path_pre="data/"):
    relations = load_file("{}/trex_multihop_relations.jsonl".format(data_path_pre))
    data_path_pre += "trex_multihop/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post
 
def get_TREx_2i_parameters(data_path_pre="data/"):
    relations = load_file("{}/trex_2i_relations.jsonl".format(data_path_pre))
    data_path_pre += "trex_2i/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post
 
def get_GoogleRE_parameters():
    relations = [
        {
            "relation": "place_of_birth",
            "template": "[X] was born in [Y] .",
            "template_negated": "[X] was not born in [Y] .",
        },
        {
            "relation": "date_of_birth",
            "template": "[X] (born [Y]).",
            "template_negated": "[X] (not born [Y]).",
        },
        {
            "relation": "place_of_death",
            "template": "[X] died in [Y] .",
            "template_negated": "[X] did not die in [Y] .",
        },
    ]
    data_path_pre = "data/Google_RE/"
    data_path_post = "_test.jsonl"
    return relations, data_path_pre, data_path_post


def get_ConceptNet_parameters(data_path_pre="data/"):
    #relations = [{"relation": "test"}]
    #data_path_pre += "ConceptNet/"
    #data_path_post = ".jsonl"
    relations = load_file("{}ConceptNet_reformat/relations.jsonl".format(data_path_pre))
    data_path_pre += 'ConceptNet_reformat/data/'
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post

def get_Squad_parameters(data_path_pre="data/"):
    relations = [{"relation": "test"}]
    data_path_pre += "Squad/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post

def run_all_LMs(parameters, d_n, given_args = None):
    args = given_args

    res_d = {}
    for ip in LMs:
        print(ip["label"])
        m_id = ip['label']
        
        args.save_dir = args.save_dir_root + '/' + d_n + '_' + m_id + '/'

        os.system('mkdir -p ' + args.save_dir)
        if args.dry_run == True:
            log_fn = args.save_dir + '/' + 'log.txt'
        else:
            log_fn = args.save_dir + '/' + 'log_dryrun.txt'
        setLogger(logger, log_fn)
     
        set_seed(args)
        res = run_experiments(*parameters, input_param=ip, use_negated_probes=False, given_args = given_args)
        res_d[m_id] = res

        print('for', m_id)
        all_res_d, mean_p1, mean_mrr, all_Precision1, type_Precision1, type_count, best_temp_d = res
        if given_args.fewshot_ft > 0 and given_args.fewshot_ft_mode == 'each':
            dev_losses = [res['fewshot_ft_res']['best_dev_loss'] for res in all_res_d.values()]
            logger.info('fewshot_ft: mean dev_loss: %f', np.mean(dev_losses))
            if given_args.fewshot_ft_tuneseed_num > 1:
                tuned_std = [np.std(res['fewshot_ft_res']['tuned_devloss_lis']) for res in all_res_d.values()]
                logger.info('tuneseed_num %d mean-std: %f', given_args.fewshot_ft_tuneseed_num, np.mean(tuned_std))

        print("@@@ mean P@1: {} mean MRR: {}".format(mean_p1, mean_mrr))
        for t, l in type_Precision1.items():
            print("@@@ type:",
                t,
                np.mean(l),
                #normweighted_sum(l, type_count[t]), #statistics.mean(l),
                sum(type_count[t]),
                len(type_count[t]),
                flush=True,
            )
    
        #save_fn = args.save_dir + '/fsctx{}_fsft{}mode{}batchmode{}_param{}_lr{}bz{}devnum{}_p1datanum{}alllrover{}_relveclr{}wd{}num{}_maxnum{}_position{}_inimode{}_scale{}_inputmode{}_tuneseednum{}_mode{}'.format(args.fewshot_context, args.fewshot_ft, args.fewshot_ft_mode, args.fewshot_ft_batch_mode, args.fewshot_ft_param_mode, args.learning_rate, args.fewshot_ft_bz, args.fewshot_ft_devnum, args.fewshot_ft_phase1_data_num, str(args.fewshot_ft_param_all_lr_overwrite), args.relvec_learning_rate, args.relvec_weight_decay, args.relvec_num, args.relvec_max_num, args.relvec_position, args.relvec_initialize_mode, args.relvec_initialize_scale, args.relvec_input_mode, args.fewshot_ft_tuneseed_num, args.fewshot_ft_tune_mode) + '_all_res.save'
        #print('saving all_res_d to', save_fn)
        if args.only_run_num <= 0:
            save_fn = 'lpaqa_prompts/templates_{}_tune{}.save'.format(m_id, args.lpaqa_tune_num)
            print('saving best_temp_d to', save_fn)
            torch.save(best_temp_d, save_fn)

    print('re-print args', str(args))

if __name__ == "__main__":
    #handling ctrl-c, has nothing to do with research
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, exit_gracefully)
    #print("1. Google-RE")
    #parameters = get_GoogleRE_parameters()
    #run_all_LMs(parameters)
   
    if 'trex' == args.dataset:
        print("2. T-REx")
        parameters = get_TREx_parameters()
        run_all_LMs(parameters, 'trex', given_args = args)
    
    if 'trex_multihop' == args.dataset:
        print("T-Rex multi-hop")
        parameters = get_TREx_multihop_parameters()
        run_all_LMs(parameters, 'trex_multihop', given_args = args)

    if 'trex_2i' == args.dataset:
        print("T-Rex 2i")
        parameters = get_TREx_2i_parameters()
        run_all_LMs(parameters, 'trex_2i', given_args = args)

    if 'conceptnet' == args.dataset:
        print("3. ConceptNet")
        parameters = get_ConceptNet_parameters()
        run_all_LMs(parameters, 'conceptnet', given_args = args)
    
    if 'squad' == args.dataset:
        print("4. SQuAD")
        parameters = get_Squad_parameters()
        run_all_LMs(parameters, 'squad', given_args = args)
    
    logger.info('%s', str(args))
