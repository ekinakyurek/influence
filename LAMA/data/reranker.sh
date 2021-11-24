#!/bin/bash


uri_file=TREx_lama_templates/abstracts/all_used_uris.txt
abstract_file=TREx_lama_templates/abstracts/all_used.jsonl
hashmap_file=TREx_lama_templates/abstracts/hashmap_used.json
test_file=TREx_lama_templates/all.tfrecord

metric_output_file=metrics/bm25/gradmetrics_bm25plus_new.json
reranker_metric_prefix=metrics/reranker/bm2plus_multi
nn_output_file=nns/bm25/bm25plus_nn_results.jsonl


T5_PREFIX=T5_checkpoints/1000000/model/pytorch_model_
CUDA_VISIBLE_DEVICES=9,10,11,12,14

# source /raid/lingo/akyurek/gitother/fewshot_lama/setup.sh

# python -u evaluate.py \
# --abstract_uri_list ${uri_file} \
# --abstract_file ${abstract_file} \
# --test_data ${test_file}  \
# --hashmap_file ${hashmap_file} \
# --nn_list_file ${nn_output_file} \
# --output_file ${metric_output_file}

# deactivate

#  --checkpoint_folders ${T5_PREFIX}5100.bin,${T5_PREFIX}10200.bin,${T5_PREFIX}15300.bin,${T5_PREFIX}111900.bin,${T5_PREFIX}1000000.bin \
eval "$(conda shell.bash hook)"
conda activate transformers

output_metric_prefix=${reranker_metric_prefix}_eos_corrects
python -u reranker.py \
  --metrics_file ${metric_output_file} \
  --hashmap_file ${hashmap_file} \
  --checkpoint_folders  ${T5_PREFIX}5100.bin,${T5_PREFIX}10200.bin,${T5_PREFIX}15300.bin,${T5_PREFIX}20400.bin,${T5_PREFIX}1000000.bin \
  --output_metrics_prefix ${output_metric_prefix} \
  --include_eos \
  --only_corrects > ${output_metric_prefix}.log 2> ${output_metric_prefix}.err

output_metric_prefix=${reranker_metric_prefix}_no_eos_corrects
python -u reranker.py \
  --metrics_file ${metric_output_file} \
  --hashmap_file ${hashmap_file} \
  --checkpoint_folders  ${T5_PREFIX}5100.bin,${T5_PREFIX}10200.bin,${T5_PREFIX}15300.bin,${T5_PREFIX}20400.bin,${T5_PREFIX}1000000.bin \
  --output_metrics_prefix ${output_metric_prefix} \
  --only_corrects > ${output_metric_prefix}.log 2> ${output_metric_prefix}.err

output_metric_prefix=${reranker_metric_prefix}_eos
python -u reranker.py \
  --metrics_file ${metric_output_file} \
  --hashmap_file ${hashmap_file} \
  --checkpoint_folders ${T5_PREFIX}5100.bin,${T5_PREFIX}10200.bin,${T5_PREFIX}15300.bin,${T5_PREFIX}20400.bin,${T5_PREFIX}1000000.bin \
  --output_metrics_prefix ${output_metric_prefix} \
  --include_eos > ${output_metric_prefix}.log 2> ${output_metric_prefix}.err

output_metric_prefix=${reranker_metric_prefix}_no_eos
python -u reranker.py \
  --metrics_file ${metric_output_file} \
  --hashmap_file ${hashmap_file} \
  --checkpoint_folders ${T5_PREFIX}5100.bin,${T5_PREFIX}10200.bin,${T5_PREFIX}15300.bin,${T5_PREFIX}20400.bin,${T5_PREFIX}1000000.bin \
  --output_metrics_prefix ${output_metric_prefix} > ${output_metric_prefix}.log 2> ${output_metric_prefix}.err


output_metric_prefix=${reranker_metric_prefix}_eos_wrongs
python -u reranker.py \
  --metrics_file ${metric_output_file} \
  --hashmap_file ${hashmap_file} \
  --checkpoint_folders  ${T5_PREFIX}5100.bin,${T5_PREFIX}10200.bin,${T5_PREFIX}15300.bin,${T5_PREFIX}20400.bin,${T5_PREFIX}1000000.bin \
  --output_metrics_prefix ${output_metric_prefix} \
  --include_eos \
  --only_wrongs > ${output_metric_prefix}.log 2> ${output_metric_prefix}.err

output_metric_prefix=${reranker_metric_prefix}_no_eos_wrongs
python -u reranker.py \
  --metrics_file ${metric_output_file} \
  --hashmap_file ${hashmap_file} \
  --checkpoint_folders  ${T5_PREFIX}5100.bin,${T5_PREFIX}10200.bin,${T5_PREFIX}15300.bin,${T5_PREFIX}20400.bin,${T5_PREFIX}1000000.bin \
  --output_metrics_prefix ${output_metric_prefix} \
  --only_wrongs > ${output_metric_prefix}.log 2> ${output_metric_prefix}.err

conda deactivate
