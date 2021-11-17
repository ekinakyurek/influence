#!/bin/bash
abstract_file=TREx_lama_templates/abstracts/all_used.tfrecord


python target_normalize.py \
  --abstract_file ${abstract_file} \
  --abstract_vectors TREx_lama_templates/encodings/train/labeled_data.tfr@100 \
  --test_file TREx_lama_templates/all.tfrecord \
  --test_vectors TREx_lama_templates/encodings/test \
  --output_file TREx_lama_templates/encodings/train/labeled_data_target_normalizedv1.tfr@100 \
  --test_output_file TREx_lama_templates/encodings/test_labeled_data_target_normalizedv1.tfr@1 \
  --gpu_workers 1
