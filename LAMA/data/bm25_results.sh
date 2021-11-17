#!/bin/bash

uri_file=TREx_lama_templates/abstracts/all_used_uris.txt
abstract_file=TREx_lama_templates/abstracts/all_used.tfrecord
hashmap_file=hashmap.json
test_file=TREx_lama_templates/all.tfrecord
abstract_json_file=TREx_lama_templates/abstracts/all_used.jsonl
nn_output_file=nns/bm25/bm25plus_nn_results.jsonl
metric_output_file=metrics/bm25/bm25plus_metrics.json


python get_nns_bm25.py \
  --abstract_file ${abstract_file} \
  --test_file ${test_file} \
  --output_file ${nn_output_file} \
  --topk 100 \

python evaluate.py \
  --abstract_uri_list ${uri_file} \
  --abstract_file ${abstract_json_file} \
  --test_data ${test_file}  \
  --hashmap_file ${hashmap_file} \
  --nn_list_file ${nn_output_file} \
  --output_file ${metric_output_file}
