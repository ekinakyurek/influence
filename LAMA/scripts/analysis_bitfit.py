import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math

fn_d = {
        10: './experiments/base//trex_roberta_large//templama_fsctx-1_fsft5modeeachbatchmodefull_batch_parambitfit_lr0.001bz8devnum5_p1datanum-1alllroverFalse_relveclr0.0001wd0.01num5_maxnum60_positionM_inimodemean_embed_scale1_inputmodedirect_tuneseednum1_modejustseed_all_res.save',
        20: './experiments/base//trex_roberta_large//templama_fsctx-1_fsft10modeeachbatchmodefull_batch_parambitfit_lr0.01bz8devnum10_p1datanum-1alllroverFalse_relveclr0.0001wd0.01num5_maxnum60_positionM_inimodemean_embed_scale1_inputmodedirect_tuneseednum1_modejustseed_all_res.save',
        40: './experiments/base//trex_roberta_large//templama_fsctx-1_fsft30modeeachbatchmodefull_batch_parambitfit_lr0.01bz8devnum10_p1datanum-1alllroverFalse_relveclr0.0001wd0.01num5_maxnum60_positionM_inimodemean_embed_scale1_inputmodedirect_tuneseednum1_modejustseed_all_res.save',
        }

ss = torch.load(fn_d[20])

diff_lis = [[] for k in range(24)]
for rel in ss:
    #print('current relation', rel)
    res = ss[rel]['fewshot_ft_res']
    for i in range(24):
        diff = math.sqrt(torch.norm(res['bitfit_params_ft'][i * 2] - res['bitfit_params_ori'][i * 2]).item() ** 2 + torch.norm(res['bitfit_params_ft'][i * 2 + 1] - res['bitfit_params_ori'][i * 2 + 1]).item() ** 2)
        diff_lis[i].append(diff)

for i in range(24):
    print(i, np.mean(diff_lis[i]))

breakpoint()
