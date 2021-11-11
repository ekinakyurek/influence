#!/bin/bash

uri_file=TREx_lama_templates/abstracts/all_used_uris.txt
abstract_file=TREx_lama_templates/abstracts/all_used.jsonl
hashmap_file=hashmap_used.json
test_file=TREx_lama_templates/all.tfrecord


nn_output_file=nn_results.jsonl
metric_output_file=gradmetrics_project_used.json

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
#  python get_nns.py \
#   --abstract_vectors TREx_lama_templates/encodings/train/labeled_data_target_normalized.tfr@100 \
#   --test_vectors TREx_lama_templates/encodings/test \
#   --output_file ${nn_output_file} \
#   --gpu_workers 4 \
#   --batch_size 100 \
#   --topk 100 \
#   --normalize

  python evaluate.py \
  --abstract_uri_list ${uri_file} \
  --abstract_file ${abstract_file} \
  --test_data ${test_file}  \
  --hashmap_file ${hashmap_file} \
  --nn_list_file ${nn_output_file} \
  --output_file ${metric_output_file}
