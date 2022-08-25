#!/bin/bash
data_root=Synth/synth_data_synth_07_27
fact_to_ids_file=${data_root}/fact_to_ids.json

nn_output_file=${data_root}/nns/bm25/unfiltered_bm25plus_nn_results_v3_hf.jsonl
metric_output_file=${data_root}/metrics/bm25/unfiltered_bm25plus_metrics_v3_hf.json

python eval/get_nns_bm25.py \
  --output_file ${nn_output_file} \
  --topk 250

# python eval/evaluate.py \
#   --fact_to_ids_file ${fact_to_ids_file} \
#   --nn_list_file ${nn_output_file} \
#   --output_file ${metric_output_file}
