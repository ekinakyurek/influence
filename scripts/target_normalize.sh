#!/bin/bash
data_root=LAMA/data
lama_root=${data_root}/TREx_lama_templates
abstract_file=${lama_root}/abstracts/all_used.tfrecord


python eval/target_normalize.py \
  --abstract_file ${abstract_file} \
  --abstract_vectors ${lama_root}/encodings/train/labeled_data.tfr@100 \
  --test_file ${lama_root}/all.tfrecord \
  --test_vectors ${lama_root}/encodings/test \
  --output_file ${lama_root}/encodings/train/labeled_data_target_normalizedv1.tfr@100 \
  --test_output_file ${lama_root}/encodings/test_labeled_data_target_normalizedv1.tfr@1 \
  --gpu_workers 1
