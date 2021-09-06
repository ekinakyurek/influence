from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
lw = list(tokenizer.get_vocab().keys())
model = RobertaForMaskedLM.from_pretrained('roberta-large')

model = model.cuda()
inputs = tokenizer("The capital of France is <mask> .", return_tensors="pt")
for k in inputs: inputs[k] = inputs[k].cuda()
input_id = inputs['input_ids'][0].tolist()
print([lw[k] for k in input_id])


labels = tokenizer("The capital of France is Paris .", return_tensors="pt")["input_ids"].cuda()
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits
print(logits[0][6][2201])

inputs = tokenizer(["The capital of France is <mask> .", "I like china and shanghai, it is a beautiful <mask>.", "I like china, it is a beautiful <mask>."], return_tensors="pt", padding = True)
for k in inputs: inputs[k] = inputs[k].cuda()
for i in range(3):
    input_id = inputs['input_ids'][i].tolist()
    print([lw[k] for k in input_id])
outputs = model(**inputs, labels=inputs['input_ids'])
logits = outputs.logits

breakpoint()
