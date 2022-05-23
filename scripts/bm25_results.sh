#!/bin/bash
data_root=LAMA/data
lama_root=${data_root}/TREx_lama_templates_v3
fact_to_ids_file=${lama_root}/abstracts/fact_to_ids_used.json

nn_output_file=${data_root}/nns/bm25/unfiltered_bm25plus_nn_results_v3_hf.jsonl
metric_output_file=${data_root}/metrics/bm25/unfiltered_bm25plus_metrics_v3_hf.json


python eval/get_nns_bm25.py \
  --output_file ${nn_output_file} \
  --topk 250

# python eval/evaluate.py \
#   --fact_to_ids_file ${fact_to_ids_file} \
#   --nn_list_file ${nn_output_file} \
#   --output_file ${metric_output_file}
