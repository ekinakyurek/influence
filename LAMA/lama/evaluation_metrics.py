# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import numpy as np
import scipy


def __max_probs_values_indices(masked_indices, log_probs, topk=1000):

    # score only first mask
    masked_indices = masked_indices[:1]

    masked_index = masked_indices[0]
    log_probs = log_probs[masked_index]

    value_max_probs, index_max_probs = torch.topk(input=log_probs,k=topk,dim=0)
    index_max_probs = index_max_probs.numpy().astype(int)
    value_max_probs = value_max_probs.detach().numpy()

    return log_probs, index_max_probs, value_max_probs


def __print_top_k(value_max_probs, index_max_probs, vocab, mask_topk, index_list, max_printouts = 10):
    result = []
    msg = "\n| Top{} predictions\n".format(max_printouts)
    for i in range(mask_topk):
        filtered_idx = index_max_probs[i].item()

        if index_list is not None:
            # the softmax layer has been filtered using the vocab_subset
            # the original idx should be retrieved
            idx = index_list[filtered_idx]
        else:
            idx = filtered_idx

        log_prob = value_max_probs[i].item()
        word_form = vocab[idx]

        if i < max_printouts:
            msg += "{:<8d}{:<20s}{:<12.3f}\n".format(
                i,
                word_form,
                log_prob
            )
        element = {'i' : i, 'token_idx': idx, 'log_prob': log_prob, 'token_word_form': word_form}
        result.append(element)
    return result, msg


def get_ranking(sample, log_probs, masked_indices, vocab, label_index = None, index_list = None, topk = 1000, P_AT = 10, print_generation=True):

    experiment_result = {}

    assert(len(label_index) <= topk)
    
    log_probs_mask = log_probs[masked_indices[0]]
    value_max_probs_all, index_max_probs_all = torch.sort(log_probs_mask, descending=True)
    log_probs, index_max_probs, value_max_probs = __max_probs_values_indices(masked_indices, log_probs, topk=topk)
    result_masked_topk, return_msg = __print_top_k(value_max_probs, index_max_probs, vocab, topk, index_list)
    experiment_result['topk'] = result_masked_topk

    if print_generation:
        print(return_msg)
    
    if index_list is None:
        index_list = list(range(200000)) #just use identity mapping

    MRR = 0.
    P_AT_X = 0.
    P_AT_1 = 0.
    PERPLEXITY = None

    if label_index is not None:
        ans_list = [vocab[idx] for idx in label_index]
        # check if the labe_index should be converted to the vocab subset
        if index_list is not None:
            for i in range(len(label_index)):
                label_index[i] = index_list.index(label_index[i])
        
        rank_list = []
        for ans_w in ans_list:
            rank = -1
            for r, max_idx in enumerate(index_max_probs_all.tolist()):
                if vocab[index_list[max_idx]] == ans_w:
                    rank = r
                    break
            assert(rank != -1)
            rank_list.append(rank)
            
        ##original code
        #query = torch.full(value_max_probs.shape, label_index, dtype=torch.long).numpy().astype(int)
        #ranking_position = (index_max_probs==query).nonzero()
        
        predict_lis = []
        for i in range(len(ans_list)):
            predict_lis.append(vocab[index_list[index_max_probs[i]]])

        # LABEL PERPLEXITY
        #TODO! fix this for multiple tokens
        tokens = torch.from_numpy(np.asarray(label_index[0]))
        label_perplexity = log_probs.gather(
            dim=0,
            index=tokens,
        )
        PERPLEXITY = label_perplexity.item()
        
        #if len(ranking_position) >= 0 and ranking_position[0].shape[0] != 0:
        #    rank = ranking_position[0][0] + 1
        mrr_lis, p_at_x_list, p_at_1_list = [], [], []
        r_c, fewshot_obj_ranks = 0, None
        if 'fewshot_obj_labels' in sample: #compensate for targets that exists few-shot learning
            r_c = len(sample['fewshot_obj_labels'])
            fewshot_obj_ranks = []
            for ans_w in sample['fewshot_obj_labels']:
                rank = -1
                for r, max_idx in enumerate(index_max_probs_all.tolist()):
                    if vocab[index_list[max_idx]] == ans_w:
                        rank = r
                        break
                assert(rank != -1)
                fewshot_obj_ranks.append(rank)
        
        """
        for rank in rank_list:
            if rank == -1:
                mpr_lis.append(0)
                p_at_x_list.append(0)
                p_at_1_list.append(0)
            else:
                rank = rank + 1
                # print("rank: {}".format(rank))
                #TODO: I NEED TO FIX THIS
                if rank <= len(ans_list) + r_c:
                    p_at_1_list.append(1)
                else:
                    p_at_1_list.append(0)
                if rank <= len(ans_list) + P_AT + r_c:
                    p_at_x_list.append(1)
                else:
                    p_at_x_list.append(0)
                rank = rank - 1 #will add back for mrr_lis
                if fewshot_obj_ranks is not None:
                    assert(rank not in fewshot_obj_ranks)
                    print('rank before:', rank)
                    rank -= sum(np.array(fewshot_obj_ranks) < rank)
                rank += 1
                mrr_lis.append(1/rank)
        assert(len(mrr_lis) == len(p_at_1_list) and len(p_at_1_list) == len(p_at_x_list))
        MRR, P_AT_X, P_AT_1 = np.mean(mrr_lis), np.mean(p_at_x_list), np.mean(p_at_1_list)
        """
        assert(100000 > P_AT * 10)
        for i in range(len(rank_list)):
            if rank_list[i] == -1:
                rank_list[i] = 100000
                mpr_lis.append(0)
            else:
                if fewshot_obj_ranks is not None:
                    assert(rank_list[i] not in fewshot_obj_ranks)
                    #print('rank before:', rank)
                    rank_list[i] -= sum(np.array(fewshot_obj_ranks) < rank_list[i])
                rank_list[i] = rank_list[i] + 1
                mrr_lis.append(1 / rank_list[i])
        MRR = np.mean(mrr_lis)
        P_AT_1 = 1.0 if (1 in rank_list) else 0.0
        P_AT_X = sum(np.array(rank_list) <= P_AT) * 1.0 / P_AT

    #if len(ans_list) > 5:
    #    breakpoint()
    assert(MRR != 0)
    experiment_result["MRR"] = MRR
    experiment_result["P_AT_X"] = P_AT_X
    experiment_result["P_AT_1"] = P_AT_1
    experiment_result["PERPLEXITY"] = PERPLEXITY
    #
    # print("MRR: {}".format(experiment_result["MRR"]))
    # print("P_AT_X: {}".format(experiment_result["P_AT_X"]))
    # print("P_AT_1: {}".format(experiment_result["P_AT_1"]))
    # print("PERPLEXITY: {}".format(experiment_result["PERPLEXITY"]))

    return MRR, P_AT_X, experiment_result, return_msg


def __overlap_negation(index_max_probs__negated, index_max_probs):
    # compares first ranked prediction of affirmative and negated statements
    # if true 1, else: 0
    return int(index_max_probs__negated == index_max_probs)


def get_negation_metric(log_probs, masked_indices, log_probs_negated,
                        masked_indices_negated, vocab, index_list=None,
                        topk = 1):

    return_msg = ""
    # if negated sentence present
    if len(masked_indices_negated) > 0:

        log_probs, index_max_probs, _ = \
            __max_probs_values_indices(masked_indices, log_probs, topk=topk)
        log_probs_negated, index_max_probs_negated, _ = \
            __max_probs_values_indices(masked_indices_negated,
                                       log_probs_negated, topk=topk)

        # overlap btw. affirmative and negated first ranked prediction: 0 or 1
        overlap = __overlap_negation(index_max_probs_negated[0],
                                     index_max_probs[0])
        # rank corrl. btw. affirmative and negated predicted log_probs
        spearman_rank_corr = scipy.stats.spearmanr(log_probs,
                                                   log_probs_negated)[0]

    else:
        overlap = np.nan
        spearman_rank_corr = np.nan

    return overlap, spearman_rank_corr, return_msg
