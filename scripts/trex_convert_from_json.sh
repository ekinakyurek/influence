#!/bin/bash
input_folder=$1
echo $input_folder

for f in ${input_folder}/*jsonl; do
    echo "$f"
    python data/json_to_tf_record.py --input_file $f
done
