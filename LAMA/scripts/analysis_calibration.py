import torch
import math
import numpy as np

def load_fn(fn):
    print('loading', fn)
    res = torch.load(fn)
    uuid_all = {}
    probs_right, probs_wrong = [], []
    for r_id in res:
        for item in res[r_id]['list_of_results']:
            #res_1['P19']['list_of_results'][0]['uuid']
            uuid = r_id + '_' + item['sample']['masked_sentence_ori'].strip().replace(' ', '_')
            if uuid in uuid_all:
                print('duplicate uuid!')
                breakpoint()
            uuid_all[uuid] = item
            predict_prob = math.exp(item['masked_topk']['topk'][0]['log_prob'])
            if item['masked_topk']['topk'][0]['token_word_form'] in item['sample']['obj_labels']:
                probs_right.append(predict_prob)
            else:
                probs_wrong.append(predict_prob)
    
    print('probs_right len:', len(probs_right), 'mean:', np.mean(probs_right), 'std:', np.std(probs_right))
    print('probs_wrong len:', len(probs_wrong), 'mean:', np.mean(probs_wrong), 'std:', np.std(probs_wrong))

    return uuid_all

#fn_1 = './experiments/base//trex_roberta_large//fsctx-1_fsft-1modeeachlr1e-06bz8devnum10_all_res.save'
#fn_1 = './experiments/base//trex_roberta_large//fsctx20_fsft-1modeeachlr1e-06bz8_all_res.save'
#fn_1 = './experiments/base/trex_roberta_large/fsctx-1_fsft20modeeachlr1e-06bz8devnum10_all_res.save' 

#fn_2 = './experiments/base//trex_roberta_large//fsctx20_fsft-1modeeachlr1e-06bz8_all_res.save'
fn_1 = './experiments/base/trex_roberta_large/fsctx-1_fsft50modeeachlr1e-06bz8devnum10_all_res.save' 

load_fn(fn_1)

