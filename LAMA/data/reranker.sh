#!/bin/bash


uri_file=TREx_lama_templates/abstracts/all_used_uris.txt
abstract_file=TREx_lama_templates/abstracts/all_used.jsonl
hashmap_file=TREx_lama_templates/abstracts/hashmap_used.json
test_file=TREx_lama_templates/all.tfrecord

metric_output_file=metrics/bm25/gradmetrics_bm25plus_new.json
reranker_metric_prefix=metrics/reranker/reranker_on_bm25plus
nn_output_file=nns/bm25/sbm25plus_nn_results.jsonl


T5_PREFIX=T5_checkpoints/1000000/model/pytorch_model_
CUDA_VISIBLE_DEVICES=0,1,2,3,4

source /raid/lingo/akyurek/gitother/fewshot_lama/setup.sh

python -u evaluate.py \
--abstract_uri_list ${uri_file} \
--abstract_file ${abstract_file} \
--test_data ${test_file}  \
--hashmap_file ${hashmap_file} \
--nn_list_file ${nn_output_file} \
--output_file ${metric_output_file}

deactivate

eval "$(conda shell.bash hook)"
conda activate transformers
python -u reranker.py \
  --metrics_file ${metric_output_file} \
  --hashmap_file ${hashmap_file} \
  --checkpoint_folders ${T5_PREFIX}5100.bin,${T5_PREFIX}10200.bin,${T5_PREFIX}15300.bin,${T5_PREFIX}111900.bin \
  --output_metrics_prefix ${reranker_metric_prefix} \
  --normalize

conda deactivate
