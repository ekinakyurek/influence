export T5=T5_checkpoints/1000000/model/

# deactivate
#
# conda activate transformers

for i in 20400; do
	transformers-cli convert \
			 --model_type t5 \
			 --tf_checkpoint $T5/model.ckpt-${i} \
			 --config $T5/config.json \
			 --pytorch_dump_output $T5/pytorch_model_${i}.bin
done
