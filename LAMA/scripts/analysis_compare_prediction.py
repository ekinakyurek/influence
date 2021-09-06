import torch

def load_fn(fn):
    print('loading', fn)
    res = torch.load(fn)
    uuid_all, uuid_right = {}, {}
    for r_id in res:
        for item in res[r_id]['list_of_results']:
            #res_1['P19']['list_of_results'][0]['uuid']
            uuid = r_id + '_' + item['sample']['masked_sentence_ori'].strip().replace(' ', '_')
            if uuid in uuid_all:
                print('duplicate uuid!')
                breakpoint()
            uuid_all[uuid] = item
            if item['sample_Precision1'] > 0.1:
                uuid_right[uuid] = item
    
    print('len(uuid_right) / len(uuid_all):', len(uuid_right) * 1.0 / len(uuid_all))

    return uuid_all, uuid_right

def compute_overlap(uuid_all_1, uuid_right_1, uuid_all_2, uuid_right_2):
    kset1, kset2, rs1, rs2 = set(uuid_all_1.keys()), set(uuid_all_2.keys()), set(uuid_right_1.keys()), set(uuid_right_2.keys())
    k_common = kset1 & kset2
    print('len(kset1)', len(kset1), 'len(kset2)', len(kset2), 'len(kset1 & kset2)', len(k_common))
    rs1 = rs1 & k_common
    rs2 = rs2 & k_common
    print('len(rs1 & rs2) / len(rs1)', len(rs1 & rs2) * 1.0 / len(rs1))
    print('len(rs1 & rs2) / len(rs2)', len(rs1 & rs2) * 1.0 / len(rs2))
    return

#fn_1 = './experiments/base//trex_roberta_large//fsctx-1_fsft-1modeeachlr1e-06bz8devnum10_all_res.save'
fn_1 = './experiments/base//trex_roberta_large//fsctx20_fsft-1modeeachlr1e-06bz8_all_res.save'
#fn_1 = './experiments/base/trex_roberta_large/fsctx-1_fsft20modeeachlr1e-06bz8devnum10_all_res.save' 

#fn_2 = './experiments/base//trex_roberta_large//fsctx20_fsft-1modeeachlr1e-06bz8_all_res.save'
fn_2 = './experiments/base/trex_roberta_large/fsctx-1_fsft50modeeachlr1e-06bz8devnum10_all_res.save' 

uuid_all_1, uuid_right_1 = load_fn(fn_1)
uuid_all_2, uuid_right_2 = load_fn(fn_2)

compute_overlap(uuid_all_1, uuid_right_1, uuid_all_2, uuid_right_2)

