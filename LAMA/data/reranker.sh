#!/bin/bash
lama_root='TREx_lama_templates_v2'

uri_file=${lama_root}/abstracts/all_used_uris.txt
hashmap_file=${lama_root}/abstracts/hashmap_used.json
test_file=${lama_root}/all.tfrecord
abstract_file=${lama_root}/abstracts/all_used.jsonl

nn_output_file=nns/bm25/bm25plus_nn_results_v2.jsonl
metric_output_file=metrics/bm25/bm25plus_metrics_v2.json



T5_PREFIX=T5_checkpoints/1000000/model/pytorch_model_
CUDA_VISIBLE_DEVICES=8,9,10,11

#source /raid/lingo/akyurek/gitother/fewshot_lama/setup.sh

# python -u evaluate.py \
# --abstract_uri_list ${uri_file} \
# --abstract_file ${abstract_file} \
# --test_data ${test_file}  \
# --hashmap_file ${hashmap_file} \
# --nn_list_file ${nn_output_file} \
# --output_file ${metric_output_file}

# deactivate

#checkpoint_folders=${T5_PREFIX}5100.bin,${T5_PREFIX}10200.bin,${T5_PREFIX}15300.bin,${T5_PREFIX}1000000.bin

checkpoint_folders=${T5_PREFIX}5100.bin,${T5_PREFIX}10200.bin,${T5_PREFIX}15300.bin,${T5_PREFIX}1000000.bin
eval "$(conda shell.bash hook)"
conda activate transformers

for eos in "eos" "no_eos"; do
  for subset in "learned" "corrects";do    
      output_metric_prefix=metrics/reranker/bm25plusv2_4ckpt_mean_multi_${eos}_${target}_${subset}
      params=("--metrics_file=${metric_output_file}" "--hashmap_file=${hashmap_file}" "--checkpoint_folders=${checkpoint_folders}" "--output_metrics_prefix=${output_metric_prefix}")
      [[ $eos == "eos" ]] && params+=(--include_eos)
      [[ $subset == "corrects" ]] && params+=(--only_corrects)
      [[ $subset == "wrongs" ]] && params+=(--only_wrongs)
      [[ $subset == "learned" ]] && params+=(--only_learned)
      python -u reranker.py "${params[@]}"  > ${output_metric_prefix}.log 2> ${output_metric_prefix}.err
    done
done

conda deactivate
