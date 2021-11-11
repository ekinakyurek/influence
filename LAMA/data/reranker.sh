#!/bin/bash

uri_file=TREx_lama_templates/abstracts/all_used_uris.txt
abstract_file=TREx_lama_templates/abstracts/all_used.tfrecord
hashmap_file=hashmap.json
test_file=TREx_lama_templates/all.tfrecord
metric_output_file=gradmetrics_encoder.json
hashmap_file=hashmap.json

CUDA_VISIBLE_DEVICES=2 \
python -u reranker.py \
  --metrics_file ${metric_output_file} \
  --hashmap_file ${hashmap_file} \
  --checkpoint_prefix './T5_checkpoints/1000000/model/pytorch_model.bin/' \
  --normalize
