#!/bin/bash

data_root=LAMA/data/
nn_output_file=${data_root}/nns/bm25/unfiltered_bm25plus_nn_results_v3_hf.jsonl
metric_output_file=${data_root}/metrics/bm25/unfiltered_bm25plus_metrics_v3_hf.pickle
T5_PREFIX=${data_root}/T5_checkpoints/1000000/model/pytorch_model_
checkpoint_folders="${T5_PREFIX}5100.bin,${T5_PREFIX}10200.bin,${T5_PREFIX}15300.bin,${T5_PREFIX}1000000.bin"

source /afs/csail.mit.edu/u/a/akyurek/akyurek/gitother/fewshot_lama/trex/bin/activate
export PYTHONPATH="/raid/lingo/akyurek/gitother/fewshot_lama"
exp_folder=LAMA/data/metrics/reranker/sweep_v2/


for weight in 0.1 0.3 0.5 0.7 0.9;
do
    weight_exp_folder=${exp_folder}/ensemble_bm25/weight_${weight}/
    mkdir -p ${weight_exp_folder}
    python ensemble.py \
        --checkpoint_folders ${checkpoint_folders} \
        --baseline_metrics_file ${metric_output_file} \
        --baseline_nn_file ${nn_output_file} \
        --data_root ${data_root} \
        --fact_to_ids_file LAMA/data/TREx_lama_templates_v3/abstracts/fact_to_ids.json \
        --baseline_reweight ${weight} \
        --gpus_to_use 0,1,2,3 \
        --load_exp_folder ${exp_folder} \
        --noeval_stage \
        --nopre_stage \
        --nockpt_stage \
        --exp_folder ${weight_exp_folder} > ${weight_exp_folder}/.log2.txt 2> ${weight_exp_folder}/.err2.txt
done
