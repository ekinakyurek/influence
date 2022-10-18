#!/bin/bash
data_root=Synth/synth_data_synth_07_27
fact_to_ids_file=${data_root}/fact_to_ids.json

nn_output_file=${data_root}/nns/bm25/unfiltered_bm25plus_nn_results_v3_hf.jsonl
metric_output_file=${data_root}/metrics/bm25/unfiltered_bm25plus_metrics_v3_hf.pickle
T5_PREFIX=${data_root}/checkpoints/pytorch_model_
checkpoint_folders="${T5_PREFIX}1050000.bin,${T5_PREFIX}1100000.bin,${T5_PREFIX}1150000.bin,${T5_PREFIX}1250000.bin"

source /afs/csail.mit.edu/u/a/akyurek/akyurek/gitother/fewshot_lama/trex/bin/activate
export PYTHONPATH="/raid/lingo/akyurek/gitother/fewshot_lama"
exp_folder=Synth/synth_data_synth_07_27/metrics/reranker/sweep/
mkdir -p ${exp_folder}

python reranker.py \
    --checkpoint_folders ${checkpoint_folders} \
    --baseline_metrics_file ${metric_output_file} \
    --baseline_nn_file ${nn_output_file} \
    --data_root ${data_root} \
    --fact_to_ids_file ${fact_to_ids_file} \
    --gpus_to_use 11,12,13,15 \
    --noeval_stage \
    --nopre_stage \
    --exp_folder ${exp_folder} >${exp_folder}/.log5.txt 2>${exp_folder}/.err5.txt

# exp_folder_ft=LAMA/data/metrics/reranker/sweep_v2_ft/
# mkdir -p ${exp_folder_ft}
# T5_PREFIX_FT=${data_root}/T5_checkpoints/finetune/checkpoint-
# checkpoint_folders_ft=${T5_PREFIX_FT}5000,${T5_PREFIX_FT}10000,${T5_PREFIX_FT}30000,${T5_PREFIX_FT}80000

# # Evaluate FT model in its own learned subset
# python reranker.py \
#     --checkpoint_folders ${checkpoint_folders_ft} \
#     --baseline_metrics_file ${metric_output_file} \
#     --baseline_nn_file ${nn_output_file} \
#     --data_root ${data_root} \
#     --fact_to_ids_file ${fact_to_ids_file} \
#     --gpus_to_use 12,13,14,15 \
#     --noeval_stage \
#     --exp_folder ${exp_folder_ft} >${exp_folder_ft}/.log.txt 2>${exp_folder_ft}/.err.txt

# # Evaluate PT model in FTs' learned subset
# exp_folder=LAMA/data/metrics/reranker/sweep_v2_pt/
# mkdir -p ${exp_folder}
# python reranker.py \
#     --checkpoint_folders ${checkpoint_folders} \
#     --baseline_metrics_file ${metric_output_file} \
#     --baseline_nn_file ${nn_output_file} \
#     --data_root ${data_root} \
#     --fact_to_ids_file ${fact_to_ids_file} \
#     --gpus_to_use 12,13,14,15 \
#     --load_exp_folder ${exp_folder_ft} \
#     --noeval_stage \
#     --nopre_stage \
#     --exp_folder ${exp_folder} >${exp_folder}/.log.txt 2>${exp_folder}/.err.txt

# exp_folder_ft=LAMA/data/metrics/reranker/sweep_v2_ft_pt/
# # Evaluate FT model on pretrained learned subset of MT5
# python reranker.py \
#     --checkpoint_folders ${checkpoint_folders_ft} \
#     --baseline_metrics_file ${metric_output_file} \
#     --baseline_nn_file ${nn_output_file} \
#     --data_root ${data_root} \
#     --fact_to_ids_file ${fact_to_ids_file} \
#     --gpus_to_use 12,13,14,15 \
#     --exp_folder ${exp_folder_ft} \
#     --noeval_stage \
#     --nopre_stage \
#     --load_exp_folder ${exp_folder} >${exp_folder_ft}/.log.txt 2>${exp_folder_ft}/.err.txt

# # Single checkpoint exps
# # checkpoint_folders_sp="${T5_PREFIX}1100000.bin"
# # exp_folder_sp=LAMA/data/metrics/reranker/sweep_1100000/
# # mkdir -p ${exp_folder_sp}
# # python reranker.py \
# #     --checkpoint_folders ${checkpoint_folders_sp} \
# #     --baseline_metrics_file ${metric_output_file} \
# #     --baseline_nn_file ${nn_output_file} \
# #     --data_root ${data_root} \
# #     --fact_to_ids_file LAMA/data/TREx_lama_templates_v3/abstracts/fact_to_ids.json \
# #     --gpus_to_use0,1,2,3 \
# #     --load_exp_folder ${exp_folder} \
# #     --exp_folder ${exp_folder_sp} > ${exp_folder_sp}/.log4.txt 2> ${exp_folder_sp}/.err4.txt
