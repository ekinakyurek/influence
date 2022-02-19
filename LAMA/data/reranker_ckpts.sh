#!/bin/bash

lama_root='TREx_lama_templates_v2'

uri_file=${lama_root}/abstracts/all_used_uris.txt
hashmap_file=${lama_root}/abstracts/hashmap_used.json
test_file=${lama_root}/all.tfrecord
abstract_file=${lama_root}/abstracts/all_used.jsonl
nn_output_file=nns/bm25/bm25plus_nn_results_v3_sentence_level.jsonl
metric_output_file=metrics/bm25/bm25plus_metrics_v3_sentence_level.json


# source /raid/lingo/akyurek/gitother/fewshot_lama/setup.sh

# python -u evaluate.py \
# --abstract_uri_list ${uri_file} \
# --abstract_file ${abstract_file} \
# --test_data ${test_file}  \
# --hashmap_file ${hashmap_file} \
# --nn_list_file ${nn_output_file} \
# --output_file ${metric_output_file}

# deactivate

eval "$(conda shell.bash hook)"
conda activate transformers
CUDA_VISIBLE_DEVICES=11,14,15
T5_PREFIX=T5_checkpoints/1000000/model/pytorch_model_
checkpoint_folders=${T5_PREFIX}5100.bin,${T5_PREFIX}10200.bin,${T5_PREFIX}1000000.bin

for i in 0 1 2; do
  for eos in "no_eos"; do
    for subset in "learned" "random"; do    
        for accum in "accum"; do
        # cnt=$((cnt + 1))
        # echo $cnt
        # 	if [[ $cnt -gt 2 ]]; then    
            output_metric_prefix=metrics/reranker/exp_layers_2ckpt_${i}
            mkdir -p ${output_metric_prefix}
            output_metric_prefix=${output_metric_prefix}/ln_sl_${eos}_${target}_${subset}_${accum}
            params=("--metrics_file=${metric_output_file}" "--seed=${i}" "--hashmap_file=${hashmap_file}" "--checkpoint_folders=${checkpoint_folders}" "--output_metrics_prefix=${output_metric_prefix}")
            [[ $eos == "eos" ]] && params+=(--include_eos)
            [[ $subset == "corrects" ]] && params+=(--only_corrects)
            [[ $subset == "wrongs" ]] && params+=(--only_wrongs)
            [[ $subset == "learned" ]] && params+=(--only_learned)
            [[ $accum == "accum" ]] && params+=(--load_accums)
            echo "Logging to ${output_metric_prefix}"
            echo "Params: "
            echo "${params[@]}"
            python -u reranker_local_norm.py "${params[@]}"  > ${output_metric_prefix}.log 2> ${output_metric_prefix}.err
          # fi
        done
    done
  done
done

for i in 0 1 2; do
  for eos in "no_eos"; do
    for subset in "learned" "random"; do      
      for accum in "accum"; do
        output_metric_prefix=metrics/reranker/exp_layers_2ckpt_${i}/gn_sl_${eos}_${target}_${subset}_${accum}
        params=("--metrics_file=${metric_output_file}" "--seed=${i}" "--hashmap_file=${hashmap_file}" "--checkpoint_folders=${checkpoint_folders}" "--output_metrics_prefix=${output_metric_prefix}")
        [[ $eos == "eos" ]] && params+=(--include_eos)
        [[ $subset == "corrects" ]] && params+=(--only_corrects)
        [[ $subset == "wrongs" ]] && params+=(--only_wrongs)
        [[ $subset == "learned" ]] && params+=(--only_learned)
        [[ $accum == "accum" ]] && params+=(--load_accums)
        echo "Logging to ${output_metric_prefix}"
        echo "Params: "
        echo "${params[@]}"
        python -u reranker.py "${params[@]}"  > ${output_metric_prefix}.log 2> ${output_metric_prefix}.err
      done
    done
  done
done
conda deactivate
