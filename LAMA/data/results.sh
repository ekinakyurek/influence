#!/bin/bash

uri_file=TREx_lama_templates/abstracts/all_used_uris.txt
abstract_file=TREx_lama_templates/abstracts/all_used.jsonl
hashmap_file=TREx_lama_templates/abstracts/hashmap_used.json
test_file=TREx_lama_templates/all.tfrecord


nn_output_file=nns/activations/nn_results_activations.jsonl
metric_output_file=metrics/activations/activation_metrics.json

source /raid/lingo/akyurek/gitother/fewshot_lama/setup.sh
CUDA_VISIBLE_DEVICES=12

 python get_nns.py \
  --abstract_vectors TREx_lama_templates/encodings/activations/train/labeled_data.tfr@100 \
  --test_vectors TREx_lama_templates/encodings/activations/test/labeled_data.tfr@1 \
  --output_file ${nn_output_file} \
  --gpu_workers 1 \
  --batch_size 100 \
  --topk 100 \
  --feature_size 3072

  python evaluate.py \
  --abstract_uri_list ${uri_file} \
  --abstract_file ${abstract_file} \
  --test_data ${test_file}  \
  --hashmap_file ${hashmap_file} \
  --nn_list_file ${nn_output_file} \
  --output_file ${metric_output_file}
