data_root=LAMA/data
lama_root=${data_root}/TREx_lama_templates_v3
export T5=${data_root}/T5_checkpoints/1000000/model/

# deactivate
#
# conda activate transformers

for i in 5100 10200 15300 20400 1000000; do
	transformers-cli convert \
			 --model_type t5 \
			 --tf_checkpoint $T5/model.ckpt-${i} \
			 --config $T5/config.json \
			 --load_accumulators_instead \
			 --pytorch_dump_output $T5/pytorch_accum_${i}.bin
done
