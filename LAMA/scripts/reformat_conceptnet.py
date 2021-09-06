import time, sys
from random import shuffle
import os, json
from collections import defaultdict

ADD_TEMPLATE = False

def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

all_d = defaultdict(list)
load_fn = 'data/ConceptNet/test.jsonl'
print('loading', load_fn)
ori_data = load_file(load_fn)
for item in ori_data:
    all_d[item['pred']].append(item)

template_d = {
    'HasSubevent': 'Something you might do while [X] is [Y] .',
    'MadeOf': '[X] is made of [Y] .',
    'HasPrerequisite': '[X] requires [Y] .',
    'MotivatedByGoal': '[X] is motivated by [Y] .',
    'AtLocation': '[X] is usually found in [Y] .',
    'CausesDesire': '[X] makes you want to [Y] .',
    'IsA': '[X] is a [Y] .',
    'NotDesires': '[X] does not want [Y] .',
    'Desires': '[X] wants [Y] .',
    'CapableOf': '[X] can [Y] .',
    'PartOf': '[X] is part of [Y] .',
    'HasA': '[X] has [Y] .',
    'UsedFor': '[X] is used for [Y] .',
    'ReceivesAction': '[X] can be [Y] .',
    'Causes': '[X] causes [Y] .',
    'HasProperty': '[X] can be [Y] .',
}

print('relations:', all_d.keys())
out_fn = 'data/ConceptNet_reformat/relations.jsonl'
print('printing catalog to', out_fn)
print('ADD_TEMPLATE', ADD_TEMPLATE)
fout = open(out_fn, 'w')
for r in all_d:
    item = {"relation": r}
    if ADD_TEMPLATE == True:
        item['template'] = template_d[r]
    jstr = json.dumps(item)
    fout.write(jstr + '\n')
fout.close()

for r in all_d:
    out_fn = 'data/ConceptNet_reformat/data/' + r + '.jsonl'
    print('printing to', out_fn)
    fout = open(out_fn, 'w')
    for item in all_d[r]:
        jstr = json.dumps(item)
        fout.write(jstr + '\n')
    fout.close()

#breakpoint()
