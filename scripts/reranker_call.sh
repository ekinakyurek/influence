#!/bin/bash

data_root=LAMA/data/
nn_output_file=${data_root}/nns/bm25/unfiltered_bm25plus_nn_results_v3_sentence_level.jsonl
metric_output_file=${data_root}/metrics/bm25/unfiltered_bm25plus_metrics_v3_sentence_level.pickle
T5_PREFIX=${data_root}/T5_checkpoints/1000000/model/pytorch_model_
checkpoint_folders="${T5_PREFIX}5100.bin,${T5_PREFIX}10200.bin,${T5_PREFIX}15300.bin,${T5_PREFIX}1000000.bin"

source /afs/csail.mit.edu/u/a/akyurek/akyurek/gitother/fewshot_lama/trex/bin/activate
export PYTHONPATH="/raid/lingo/akyurek/gitother/fewshot_lama"
# CUDA_VISIBLE_DEVICES=13
exp_folder=LAMA/data/metrics/reranker/sweep/
mkdir -p ${exp_folder}

python reranker.py \
    --checkpoint_folders ${checkpoint_folders} \
    --baseline_metrics_file ${metric_output_file} \
    --baseline_nn_file ${nn_output_file} \
    --data_root ${data_root} \
    --lama_folder LAMA/data/TREx_lama_templates_v3 \
    --gpus_to_use 2,3,14,15\
    --exp_folder ${exp_folder} > ${exp_folder}/.log.txt 2> ${exp_folder}/.err.txt
