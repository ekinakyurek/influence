#!/bin/bash
input_folder=$1
total_shards=$(ls ${input_folder}/*jsonl | wc -l)
# shard_no=0
echo $input_folder

# cat ${input_folder}/*jsonl > ${input_folder}/all.jsonl

for f in ${input_folder}/P*jsonl; do
    echo "$f"
    echo "total files: ${total_shards}"
#   echo "shard no: ${shard_no}"
    python to_tf_record.py  --input_file $f
#   shard_no=$(($shard_no + 1))
done

#python extract_abstracts.py --input_folder ${input_folder} --abstract_file original_trex/all.jsonl
