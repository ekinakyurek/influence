#!/bin/bash
data_root=LAMA/data
lama_root=${data_root}TREx_lama_templates_v3
uri_file=${lama_root}/abstracts/all_used_uris.txt
abstract_file=${lama_root}/abstracts/all_used.tfrecord
hashmap_file=${lama_root}/abstracts/hashmap_used.json
test_file=${lama_root}/all.tfrecord
abstract_json_file=${lama_root}/abstracts/all_used.jsonl

nn_output_file=${data_root}/nns/bm25/unfiltered_bm25plus_nn_results_v3_sentence_level.jsonl
metric_output_file=${data_root}/metrics/bm25/unfiltered_bm25plus_metrics_v3_sentence_level.json


python eval/get_nns_bm25.py \
  --abstract_file ${abstract_file} \
  --test_file ${test_file} \
  --output_file ${nn_output_file} \
  --topk 250

# python evaluate.py \
#   --abstract_uri_list ${uri_file} \
#   --abstract_file ${abstract_json_file} \
#   --test_data ${test_file}  \
#   --hashmap_file ${hashmap_file} \
#   --nn_list_file ${nn_output_file} \
#   --output_file ${metric_output_file}
