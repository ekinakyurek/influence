import torch
import transformers
from transformers import MT5ForConditionalGeneration


def load_model_with_accum(checkpoint_folder, gpu):
    model = MT5ForConditionalGeneration.from_pretrained(
        checkpoint_folder, local_files_only=True
    ).cuda(gpu)

    model.accums = {}

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=4e-5)
    optim_dict = torch.load(f"{checkpoint_folder}/optimizer.pt")

    optimizer.load_state_dict(optim_dict)

    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state[p]
            exp_avg_sq = state["exp_avg_sq"]
            print(group["eps"])
            denom = exp_avg_sq.sqrt().add_(1e-7)
            p.accum = denom.cuda(gpu)

    for name, p in model.named_parameters():
        model.accums[name] = p.accum.flatten()
        p.accum = None

    return model
