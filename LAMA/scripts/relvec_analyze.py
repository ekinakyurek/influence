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

roberta_con = torch.load('pre-trained_language_models/roberta_model_20210426.save')

res_ld = torch.load('./experiments/base//trex_roberta_large//templama_fsctx-1_fsft30modeeachbatchmodefull_batch_paramonly_relvec_lr1e-06bz8devnum10_p1datanum-1alllroverFalse_relveclr0.01wd0.01num5_maxnum60_positionM_inimodemean_embed_scale1_inputmodedirect_tuneseednum40_modejustseed_all_res.save')

word_embeds = roberta_con.model.roberta.embeddings.word_embeddings.weight

vocab = roberta_con.vocab

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

for rel in res_ld:
    print(rel)
    print(res_ld[rel]['relation'])
    relvecs = res_ld[rel]['relvec_save'].data
    for i in range(5):
        relvec = relvecs[i]
        #dis = cos(relvec.repeat(word_embeds.size(0), 1), word_embeds)
        dis = torch.norm(relvec.repeat(word_embeds.size(0), 1) - word_embeds, dim = 1)
        #dis[4] = +100
        #dis[6], dis[5], dis[2], dis[8], dis[50264] = 100, 100, 100, 100, 100
        dis[:50] = 100
        dis[-10000:] = 100
        w_id = torch.argmin(dis).item()
        print(i, vocab[w_id], w_id)

breakpoint()
