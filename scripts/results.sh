#!/bin/bash

data_root=LAMA/data
lama_root=${data_root}/TREx_lama_templates

uri_file=${lama_root}/abstracts/all_used_uris.txt
abstract_file=${lama_root}/abstracts/all_used.jsonl
hashmap_file=${lama_root}/abstracts/hashmap_used.json
test_file=${lama_root}/all.tfrecord


nn_output_file=${data_root}/nns/activations/nn_results_activations.jsonl
metric_output_file=${data_root}/metrics/activations/activation_metrics.json

source /raid/lingo/akyurek/gitother/fewshot_lama/setup.sh
CUDA_VISIBLE_DEVICES=12

 python eval/get_nns.py \
  --abstract_vectors ${lama_root}/encodings/activations/train/labeled_data.tfr@100 \
  --test_vectors ${lama_root}/encodings/activations/test/labeled_data.tfr@1 \
  --output_file ${nn_output_file} \
  --gpu_workers 1 \
  --batch_size 100 \
  --topk 100 \
  --feature_size 3072

  python eval/evaluate.py \
  --abstract_uri_list ${uri_file} \
  --abstract_file ${abstract_file} \
  --test_data ${test_file}  \
  --hashmap_file ${hashmap_file} \
  --nn_list_file ${nn_output_file} \
  --output_file ${metric_output_file}
