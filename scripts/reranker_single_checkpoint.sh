#!/bin/bash
data_root=LAMA/data/
metric_output_file=${data_root}/metrics/bm25/unfiltered_bm25plus_metrics_v3_hf.pickle
source /afs/csail.mit.edu/u/a/akyurek/akyurek/gitother/fewshot_lama/trex/bin/activate
export PYTHONPATH="/raid/lingo/akyurek/gitother/fewshot_lama"

for ckpt_no in 5000 10000 30000 80000; do
    exp_folder="LAMA/data/metrics/reranker/sweep_re_ft_fl/"
    exp_folder_sp="LAMA/data/metrics/reranker/sweep_re_ft_fl_${ckpt_no}/"
    mkdir -p ${exp_folder_sp}
    T5_PREFIX_FT=${data_root}/T5_checkpoints/finetune/checkpoint-
    checkpoint_folders_sp=${T5_PREFIX_FT}${ckpt_no}

    python reranker.py \
        --checkpoint_folders ${checkpoint_folders_sp} \
        --baseline_metrics_file ${metric_output_file} \
        --baseline_nn_file ${nn_output_file} \
        --data_root ${data_root} \
        --ckpt_score_prefix=LAMA/data/metrics/reranker/sweep_re_ft_fl/ \
        --fact_to_ids_file LAMA/data/TREx_lama_templates_v3/abstracts/fact_to_ids.json \
        --gpus_to_use 12 \
        --load_exp_folder ${exp_folder} \
        --ckpt_no ${ckpt_no} \
        --noeval_stage \
        --nopre_stage \
        --nockpt_stage \
        --exp_folder ${exp_folder_sp} >${exp_folder_sp}/.log4.txt 2>${exp_folder_sp}/.err4.txt &
done
