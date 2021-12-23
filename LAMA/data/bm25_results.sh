#!/bin/bash
lama_root='TREx_lama_templates_v2'
uri_file=${lama_root}/abstracts/all_used_uris.txt
abstract_file=${lama_root}/abstracts/all_used.tfrecord
hashmap_file=${lama_root}/abstracts/hashmap_used.json
test_file=${lama_root}/all.tfrecord
abstract_json_file=${lama_root}/abstracts/all_used.jsonl

nn_output_file=nns/bm25/bm25plus_nn_results_v2_sentence_level.jsonl
metric_output_file=metrics/bm25/bm25plus_metrics_v2_sentence_level.json


python get_nns_bm25.py \
  --abstract_file ${abstract_file} \
  --test_file ${test_file} \
  --output_file ${nn_output_file} \
  --only_masked_sentence \
  --topk 100

python evaluate.py \
  --abstract_uri_list ${uri_file} \
  --abstract_file ${abstract_json_file} \
  --test_data ${test_file}  \
  --hashmap_file ${hashmap_file} \
  --nn_list_file ${nn_output_file} \
  --output_file ${metric_output_file}
