#!/bin/bash
data_root=LAMA/data
input_folder=$1
total_shards=$(ls ${input_folder}/*jsonl | wc -l)
echo $input_folder

cat ${input_folder}/*jsonl > ${input_folder}/all.jsonl

for f in ${input_folder}/P*jsonl; do
    echo "$f"
    python data/to_tf_record.py  --input_file $f
done

python data/to_tf_record.py  --input_file ${input_folder}/all.jsonl
mkdir -p ${input_folder}/abstracts/
python data/extract_abstracts.py --input_folder ${input_folder} --abstract_file ${data_root}/original_trex/all.jsonl
