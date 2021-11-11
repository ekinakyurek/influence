export T5=./model/

transformers-cli convert \
		 --model_type t5 \
		 --tf_checkpoint $T5/model.ckpt-5100 \
		 --config $T5/config.json \
		 --pytorch_dump_output $T5/pytorch_model.bin
