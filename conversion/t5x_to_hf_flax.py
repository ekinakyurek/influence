"""
import torch
from transformers import T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained("flax_dump_folder")
model2 = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-small")
model2_named_params = dict(model2.named_parameters())
[
    (torch.abs(p1 - model2_named_params[k]).mean().item(), k)
    for k, p1 in model.named_parameters()
    if (torch.abs(p1 - model2_named_params[k]).mean().item() > 1e-5)
]
"""
import argparse
import numpy as np
import torch
from t5x import checkpoints
from transformers import T5Config, T5ForConditionalGeneration


def get_param_value(value, name, mode="params"):
    if mode == "params":
        return value
    elif mode == "accums":
        if len(value["v"]) > 1:
            return value["v"]
        else:
            vr = value["v_row"]
            vc = value["v_col"]
            vr = np.expand_dims(vr, axis=1)  # [m, 1]
            vc = np.expand_dims(vc, axis=0)  # [1, n]
            v = vr.dot(vc) / np.sum(vr)  # [m, n]
            v = np.transpose(v)
            if (
                "attention_out" in name
                or "wi" in name
                or "logits_dense" in name
            ):
                v = np.transpose(v)

            return v


@torch.no_grad()
def convert_t5x_checkpoint_to_flax(
    t5x_checkpoint_path,
    config_name,
    flax_dump_folder_path,
    mode="params",
):
    config = T5Config.from_pretrained(config_name)
    torch_model = T5ForConditionalGeneration.from_pretrained(
        config_name, config=config
    )
    for name, param in torch_model.named_parameters():
        param.zero_()

    t5x_model = checkpoints.load_t5x_checkpoint(t5x_checkpoint_path)
    if mode == "params":
        t5x_model_params = t5x_model["target"]
    elif mode == "accums":
        print("Loading accumulators")
        t5x_model_params = t5x_model["state"]["param_states"]

    # Encoder
    for layer_index in range(config.num_layers):
        layer_name = f"layers_{layer_index}"

        # Self-Attention
        t5x_attention_key = get_param_value(
            t5x_model_params["encoder"][layer_name]["attention"]["key"][
                "kernel"
            ],
            "attention_key",
            mode=mode,
        )
        t5x_attention_out = get_param_value(
            t5x_model_params["encoder"][layer_name]["attention"]["out"][
                "kernel"
            ],
            "attention_out",
            mode=mode,
        )
        t5x_attention_query = get_param_value(
            t5x_model_params["encoder"][layer_name]["attention"]["query"][
                "kernel"
            ],
            "attention_query",
            mode=mode,
        )
        t5x_attention_value = get_param_value(
            t5x_model_params["encoder"][layer_name]["attention"]["value"][
                "kernel"
            ],
            "attention_value",
            mode=mode,
        )

        # Layer Normalization
        t5x_attention_layer_norm = get_param_value(
            t5x_model_params["encoder"][layer_name]["pre_attention_layer_norm"][
                "scale"
            ],
            "pre_attention_layer_norm",
            mode=mode,
        )

        # MLP
        t5x_mlp_wi_0 = get_param_value(
            t5x_model_params["encoder"][layer_name]["mlp"]["wi_0"]["kernel"],
            "wi_0",
            mode=mode,
        )
        t5x_mlp_wi_1 = get_param_value(
            t5x_model_params["encoder"][layer_name]["mlp"]["wi_1"]["kernel"],
            "wi_1",
            mode=mode,
        )
        t5x_mlp_wo = get_param_value(
            t5x_model_params["encoder"][layer_name]["mlp"]["wo"]["kernel"],
            "wo",
            mode=mode,
        )

        # Layer Normalization
        t5x_mlp_layer_norm = get_param_value(
            t5x_model_params["encoder"][layer_name]["pre_mlp_layer_norm"][
                "scale"
            ],
            "pre_mlp_layer_norm",
            mode=mode,
        )

        # Assigning
        torch_model.encoder.block[layer_index].layer[0].SelfAttention.k.weight[
            :, :
        ] = torch.from_numpy(t5x_attention_key.T)
        torch_model.encoder.block[layer_index].layer[0].SelfAttention.o.weight[
            :, :
        ] = torch.from_numpy(t5x_attention_out.T)
        torch_model.encoder.block[layer_index].layer[0].SelfAttention.q.weight[
            :, :
        ] = torch.from_numpy(t5x_attention_query.T)
        torch_model.encoder.block[layer_index].layer[0].SelfAttention.v.weight[
            :, :
        ] = torch.from_numpy(t5x_attention_value.T)

        torch_model.encoder.block[layer_index].layer[0].layer_norm.weight[
            :
        ] = torch.from_numpy(t5x_attention_layer_norm)

        torch_model.encoder.block[layer_index].layer[
            1
        ].DenseReluDense.wi_0.weight[:, :] = torch.from_numpy(t5x_mlp_wi_0.T)
        torch_model.encoder.block[layer_index].layer[
            1
        ].DenseReluDense.wi_1.weight[:, :] = torch.from_numpy(t5x_mlp_wi_1.T)
        torch_model.encoder.block[layer_index].layer[
            1
        ].DenseReluDense.wo.weight[:, :] = torch.from_numpy(t5x_mlp_wo.T)
        torch_model.encoder.block[layer_index].layer[1].layer_norm.weight[
            :
        ] = torch.from_numpy(t5x_mlp_layer_norm)

    t5x_encoder_norm = get_param_value(
        t5x_model_params["encoder"]["encoder_norm"]["scale"],
        "encoder_norm",
        mode=mode,
    )

    # Only for layer 0:
    t5x_encoder_rel_embedding = get_param_value(
        t5x_model_params["encoder"]["relpos_bias"]["rel_embedding"],
        "rel_embedding",
        mode=mode,
    )
    x, y = t5x_encoder_rel_embedding.shape

    # Assigning
    torch_model.encoder.block[0].layer[
        0
    ].SelfAttention.relative_attention_bias.weight[:, :] = torch.from_numpy(
        t5x_encoder_rel_embedding.T
    )
    torch_model.encoder.final_layer_norm.weight[:] = torch.from_numpy(
        t5x_encoder_norm
    )

    # Decoder
    for layer_index in range(config.num_layers):
        layer_name = f"layers_{layer_index}"

        # Self-Attention
        t5x_attention_key = get_param_value(
            t5x_model_params["decoder"][layer_name]["self_attention"]["key"][
                "kernel"
            ],
            "self_attention_key",
            mode=mode,
        )
        t5x_attention_out = get_param_value(
            t5x_model_params["decoder"][layer_name]["self_attention"]["out"][
                "kernel"
            ],
            "self_attention_out",
            mode=mode,
        )
        t5x_attention_query = get_param_value(
            t5x_model_params["decoder"][layer_name]["self_attention"]["query"][
                "kernel"
            ],
            "self_attention_query",
            mode=mode,
        )

        t5x_attention_value = get_param_value(
            t5x_model_params["decoder"][layer_name]["self_attention"]["value"][
                "kernel"
            ],
            "self_attention_value",
            mode=mode,
        )

        # Layer Normalization
        t5x_pre_attention_layer_norm = get_param_value(
            t5x_model_params["decoder"][layer_name][
                "pre_self_attention_layer_norm"
            ]["scale"],
            "pre_self_attention_layer_norm",
            mode=mode,
        )

        # Encoder-Decoder-Attention
        t5x_enc_dec_attention_key = get_param_value(
            t5x_model_params["decoder"][layer_name][
                "encoder_decoder_attention"
            ]["key"]["kernel"],
            "encode_decoder_attention_key",
            mode=mode,
        )
        t5x_enc_dec_attention_out = get_param_value(
            t5x_model_params["decoder"][layer_name][
                "encoder_decoder_attention"
            ]["out"]["kernel"],
            "encoder_decoder_attention_out",
            mode=mode,
        )
        t5x_enc_dec_attention_query = get_param_value(
            t5x_model_params["decoder"][layer_name][
                "encoder_decoder_attention"
            ]["query"]["kernel"],
            "encoder_decoder_attention_query",
            mode=mode,
        )
        t5x_enc_dec_attention_value = get_param_value(
            t5x_model_params["decoder"][layer_name][
                "encoder_decoder_attention"
            ]["value"]["kernel"],
            "encoder_decoder_attention_value",
            mode=mode,
        )

        # Layer Normalization
        t5x_cross_layer_norm = get_param_value(
            t5x_model_params["decoder"][layer_name][
                "pre_cross_attention_layer_norm"
            ]["scale"],
            "pre_cross_attention_layer_norm",
            mode=mode,
        )

        # MLP
        t5x_mlp_wi_0 = get_param_value(
            t5x_model_params["decoder"][layer_name]["mlp"]["wi_0"]["kernel"],
            "wi_0",
            mode=mode,
        )
        t5x_mlp_wi_1 = get_param_value(
            t5x_model_params["decoder"][layer_name]["mlp"]["wi_1"]["kernel"],
            "wi_1",
            mode=mode,
        )
        t5x_mlp_wo = get_param_value(
            t5x_model_params["decoder"][layer_name]["mlp"]["wo"]["kernel"],
            "wo",
            mode=mode,
        )

        # Layer Normalization
        tx5_mlp_layer_norm = get_param_value(
            t5x_model_params["decoder"][layer_name]["pre_mlp_layer_norm"][
                "scale"
            ],
            "pre_mlp_layer_norm",
            mode=mode,
        )

        # Assigning
        torch_model.decoder.block[layer_index].layer[0].SelfAttention.k.weight[
            :, :
        ] = torch.from_numpy(t5x_attention_key.T)
        torch_model.decoder.block[layer_index].layer[0].SelfAttention.o.weight[
            :, :
        ] = torch.from_numpy(t5x_attention_out.T)
        torch_model.decoder.block[layer_index].layer[0].SelfAttention.q.weight[
            :, :
        ] = torch.from_numpy(t5x_attention_query.T)
        torch_model.decoder.block[layer_index].layer[0].SelfAttention.v.weight[
            :, :
        ] = torch.from_numpy(t5x_attention_value.T)

        torch_model.decoder.block[layer_index].layer[0].layer_norm.weight[
            :
        ] = torch.from_numpy(t5x_pre_attention_layer_norm)

        torch_model.decoder.block[layer_index].layer[
            1
        ].EncDecAttention.k.weight[:, :] = torch.from_numpy(
            t5x_enc_dec_attention_key.T
        )
        torch_model.decoder.block[layer_index].layer[
            1
        ].EncDecAttention.o.weight[:, :] = torch.from_numpy(
            t5x_enc_dec_attention_out.T
        )
        torch_model.decoder.block[layer_index].layer[
            1
        ].EncDecAttention.q.weight[:, :] = torch.from_numpy(
            t5x_enc_dec_attention_query.T
        )
        torch_model.decoder.block[layer_index].layer[
            1
        ].EncDecAttention.v.weight[:, :] = torch.from_numpy(
            t5x_enc_dec_attention_value.T
        )

        torch_model.decoder.block[layer_index].layer[1].layer_norm.weight[
            :
        ] = torch.from_numpy(t5x_cross_layer_norm)

        torch_model.decoder.block[layer_index].layer[
            2
        ].DenseReluDense.wi_0.weight[:, :] = torch.from_numpy(t5x_mlp_wi_0.T)

        torch_model.decoder.block[layer_index].layer[
            2
        ].DenseReluDense.wi_1.weight[:, :] = torch.from_numpy(t5x_mlp_wi_1.T)

        torch_model.decoder.block[layer_index].layer[
            2
        ].DenseReluDense.wo.weight[:, :] = torch.from_numpy(t5x_mlp_wo.T)

        torch_model.decoder.block[layer_index].layer[2].layer_norm.weight[
            :
        ] = torch.from_numpy(tx5_mlp_layer_norm)

    # Decoder Normalization
    tx5_decoder_norm = get_param_value(
        t5x_model_params["decoder"]["decoder_norm"]["scale"],
        "decoder_norm",
        mode=mode,
    )
    torch_model.decoder.final_layer_norm.weight[:] = torch.from_numpy(
        tx5_decoder_norm
    )

    # Only for layer 0:
    t5x_decoder_rel_embedding = get_param_value(
        t5x_model_params["decoder"]["relpos_bias"]["rel_embedding"],
        "rel_embedding",
        mode=mode,
    )
    x, y = t5x_decoder_rel_embedding.shape

    torch_model.decoder.block[0].layer[
        0
    ].SelfAttention.relative_attention_bias.weight[:, :] = torch.from_numpy(
        t5x_decoder_rel_embedding.T
    )

    # Token Embeddings
    tx5_token_embeddings_inp = get_param_value(
        t5x_model_params["token_embedder"]["embedding"], "embedding", mode=mode
    )

    torch_model.shared.weight[:, :] = torch.from_numpy(tx5_token_embeddings_inp)
    tx5_token_embeddings_out = get_param_value(
        t5x_model_params["decoder"]["logits_dense"]["kernel"],
        "logits_dense",
        mode=mode,
    )
    torch_model.lm_head.weight[:, :] = torch.from_numpy(
        tx5_token_embeddings_out.T
    )
    torch_model.save_pretrained(flax_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--t5x_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path the TX5 checkpoint.",
    )
    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        required=True,
        help="Config name of T5 model.",
    )
    parser.add_argument(
        "--flax_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output FLAX model.",
    )
    parser.add_argument(
        "--mode", default="params", type=str, help="params vs accums to load"
    )
    args = parser.parse_args()
    convert_t5x_checkpoint_to_flax(
        args.t5x_checkpoint_path,
        args.config_name,
        args.flax_dump_folder_path,
        mode=args.mode,
    )
