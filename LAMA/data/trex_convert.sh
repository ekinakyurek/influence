#!/bin/bash
input_folder=$1
total_shards=$(ls ${input_folder}/*jsonl | wc -l)
echo $input_folder

cat ${input_folder}/*jsonl > ${input_folder}/all.jsonl

for f in ${input_folder}/P*jsonl; do
    echo "$f"
    python to_tf_record.py  --input_file $f
done

python to_tf_record.py  --input_file ${input_folder}/all.jsonl
mkdir -p ${input_folder}/abstracts/
python extract_abstracts.py --input_folder ${input_folder} --abstract_file original_trex/all.jsonl

pcregrep  -o1 --buffer-size=999999 '"sentence_uris": "(.*?)"'  ${input_folder}/abstracts/all_used.jsonl > ${input_folder}/all_used_uris.txt
pcregrep  -o1 --buffer-size=999999 '"sentence_uris": "(.*?)"'  ${input_folder}/abstracts/all.jsonl > ${input_folder}/all_uris.txt
