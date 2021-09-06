# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import pytorch_pretrained_bert.tokenization as btok
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM, BasicTokenizer, BertModel
import numpy as np
from lama.modules.base_connector import *
import torch.nn.functional as F
import torch.nn as nn

from transformers import RobertaTokenizer #, RobertaForMaskedLM
from .my_modeling_roberta import MyRobertaForMaskedLM 
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import torch, copy, random
import logging
logger = logging.getLogger()

class Roberta(Base_Connector):

    def __init__(self, args, vocab_subset = None):
        super().__init__()

        roberta_model_name = args.roberta_model_name
        assert(roberta_model_name == 'roberta-large')

        # When using a cased model, make sure to pass do_lower_case=False directly to BaseTokenizer
        do_lower_case = False
        #if 'uncased' in bert_model_name:
        #    do_lower_case=True

        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.tokenizer = tokenizer
        model = MyRobertaForMaskedLM.from_pretrained('roberta-large')
        self.model_type_name, self.args = 'roberta', args
        self.masked_model, self.model, self.mask_token, self.mask_token_id = model, model, '<mask>', tokenizer.mask_token_id
        self.f_mode = 'default'    #forward mode

        #self.relvec_tokens = ["<relvec0>", "<relvec1>", "<relvec2>", "<relvec3>", "<relvec4>", "<relvec5>", "<relvec6>", "<relvec7>", "<relvec8>", "<relvec9>"]
        if hasattr(args, 'relvec_max_num'): 
            logger.info('using relvec_max_num : %d', args.relvec_max_num)
            self.relvec_tokens = ["<relvec{}>".format(k) for k in range(args.relvec_max_num)]
            logger.info('[roberta] adding relvec_tokens to tokenizer: %s', str(self.relvec_tokens))
            self.relvec_idxs = []
            for relvec_token in self.relvec_tokens:
                tokenizer._add_tokens([relvec_token], special_tokens = True)
                self.relvec_idxs.append(len(tokenizer) - 1)
            
            logger.info('[roberta] resize embeddings, and setting them to zero, note: the output linear weight will also be set to zero, i might want to reset them everytime during finetuning')
            model.resize_token_embeddings(len(tokenizer))
            for idx in self.relvec_idxs:
                model.roberta.embeddings.word_embeddings.weight[idx] = 0
                model.lm_head.bias[idx] = -100
                #model.lm_head.decoder.weight[i] will also be se to zero because they are shared
            self.relvec_params, self.relvec_mlp = None, None
            if args.relation_mode == 'relvec':
                self.f_mode = 'relvec'
                self.reinitialize_relvec_params(args.seed)
                """
                self.relvec_params = torch.nn.Parameter(torch.randn(len(self.relvec_tokens), model.roberta.embeddings.word_embeddings.weight.size(1)).cuda() * args.relvec_initialize_scale)
                if args.relvec_input_mode == 'mlp':
                    self.relvec_mlp = self.create_relvec_mlp()
                    logger.info('relvec mlp : %s', str(self.relvec_mlp))
                """

        self.model.eval()

        # original vocab
        self.map_indices, self.vocab = None, []
        self.inverse_vocab_ori = {} #with the chr(288)
        for idx, w in enumerate(list(tokenizer.get_vocab().keys())): #list(self.tokenizer.ids_to_tokens.values())
            self.inverse_vocab_ori[w] = idx
            if w.startswith(chr(288)): w = w[1:]
            self.vocab.append(w)
        print('Warning: for vocab of Roberta, we remove the leading chr(288) for even token in its tokenizer vocab')
        print('First 100 tokens of self.vocab:', str(self.vocab[:100]))

        #self._init_inverse_vocab()

        # Add custom tokenizer to avoid splitting the ['MASK'] token
        #custom_basic_tokenizer = CustomBaseTokenizer(do_lower_case = do_lower_case)
        #self.tokenizer.basic_tokenizer = custom_basic_tokenizer

        # ... to get hidden states
        #self.bert_model = self.masked_bert_model.bert

        self.pad_id = self.tokenizer.pad_token_id

        self.unk_index = self.tokenizer.unk_token_id
    
    def create_relvec_mlp(self):
        ll, args = [], self.args
        assert(args.relvec_mlp_layer_num >= 1)
        ll = [nn.Linear(1024, args.relvec_mlp_hidden_dim), nn.Tanh()]
        for k in range(args.relvec_mlp_layer_num - 1):
            ll.extend([nn.Linear(args.relvec_mlp_hidden_dim, args.relvec_mlp_hidden_dim), nn.Tanh()])
        ll.append(nn.Linear(args.relvec_mlp_hidden_dim, 1024))
        if args.relvec_mlp_final_tanh == True:
            ll.append(nn.Tanh())
        mlp = torch.nn.Sequential(*ll).cuda()
        return mlp
    
    def reinitialize_relvec_params(self, seed, sample = None):
        logger.info('reinitialize_relvec_params with seed %d scale %f', seed, self.args.relvec_initialize_scale)
        torch.manual_seed(seed)
        model, args, tokenizer = self.model, self.args, self.tokenizer
        
        if args.relvec_initialize_mode == 'gaussian':
            self.relvec_params = torch.nn.Parameter(torch.randn(len(self.relvec_tokens), model.roberta.embeddings.word_embeddings.weight.size(1)).cuda() * self.args.relvec_initialize_scale)
        if args.relvec_initialize_mode == 'from_template':
            if sample is not None:
                ms = sample['templated_sentences'][0]
                ww = torch.zeros(len(self.relvec_tokens), model.roberta.embeddings.word_embeddings.weight.size(1)).cuda()
                input_ids_ori = torch.LongTensor(tokenizer([ms])['input_ids'])
                input_ids_relvec = self.convert_template_to_relvec(input_ids_ori, [sample]) 
                msg = ''
                for i, r_idx in enumerate(input_ids_relvec[0].tolist()):
                    if r_idx in self.relvec_idxs:
                        r = self.relvec_idxs.index(r_idx)
                        w_idx = input_ids_ori[0][i].item()
                        with torch.no_grad():
                            ww[r] = model.roberta.embeddings.word_embeddings.weight[w_idx].detach().data
                        msg += '{} initialized to {}, '.format(self.relvec_tokens[r], self.vocab[w_idx])
                logger.info('%s', msg)
                self.relvec_params = torch.nn.Parameter(ww) 
            else:
                self.relvec_params = torch.nn.Parameter(torch.zeros(len(self.relvec_tokens), model.roberta.embeddings.word_embeddings.weight.size(1)).cuda() * self.args.relvec_initialize_scale)

        if args.relvec_initialize_mode == 'mean_embed':
            logger.info('relvec_initialize_mode is mean_embed, taking mean from first 10k embeddings')
            mean_v = model.roberta.embeddings.word_embeddings.weight[10:10000, :].mean(dim = 0).detach().data
            ww = torch.randn(len(self.relvec_tokens), model.roberta.embeddings.word_embeddings.weight.size(1)).cuda()
            #self.relvec_params = torch.nn.Parameter(torch.randn(len(self.relvec_tokens), model.roberta.embeddings.word_embeddings.weight.size(1)).cuda())
            with torch.no_grad():
                for k in range(ww.size(0)):
                    ww[k] = mean_v
            self.relvec_params = torch.nn.Parameter(ww)

        if self.args.relvec_input_mode == 'mlp':
            self.relvec_mlp = self.create_relvec_mlp()

    def init_indices_for_filter_logprobs(self, vocab_subset, logger=None):
        print('building indices for filter_logprobs specially for Roberta (because each word could have two versions)')
        if hasattr(self, 'last_built_index_list'):
            index_list = self.last_built_index_list
        else:
            index_list = []
            hit_vocab = []
            for w_idx, w in enumerate(self.vocab):
                if w in vocab_subset:
                    index_list.append(w_idx)
                    hit_vocab.append(w)
            
            assert(len(set(hit_vocab)) == len(vocab_subset))
            print('len(index_list):', len(index_list), 'len(vocab_subset)', len(vocab_subset), 'the first one should be larger than the second one for roberta')
            self.last_built_index_list = index_list

        indices = torch.as_tensor(index_list)
        return indices, index_list

    def get_id(self, string):
        tokenizer = self.tokenizer
        inputs = tokenizer([string], return_tensors="pt")
        return inputs['input_ids'][0].tolist()

    def get_id_obj_label(self, string):
        tokenizer = self.tokenizer
        inputs = tokenizer([' ' + string], return_tensors="pt")
        return inputs['input_ids'][0].tolist()[1:-1] 

    def _cuda(self):
        self.model = self.model.cuda()
        self.masked_model = self.masked_model.cuda()

    def get_batch_generation(self, sentences_list, samples_list, logger= None,
                             try_cuda=True):
        if not sentences_list:
            return None
        assert(try_cuda == True)
        self.try_cuda()

        sentences_list = copy.deepcopy(sentences_list)
        for k in range(len(sentences_list)):
            assert(len(sentences_list[k]) == 1)
            sentences_list[k] = sentences_list[k][0]
            assert('[MASK]' in sentences_list[k] or '<mask>' in sentences_list[k])
            sentences_list[k] = sentences_list[k].replace('[MASK]', '<mask>')
        #if try_cuda:
        #    self.try_cuda()
        
        tokenizer, model, args = self.tokenizer, self.model, self.args
        model.eval()
        sentences_list = self.truncate_512(sentences_list)
        inputs = tokenizer(sentences_list, return_tensors="pt", padding = True)
        assert(inputs['input_ids'].size(1) <= 512)
        for k in inputs: inputs[k] = inputs[k].cuda()
        
        if args.relation_mode == 'relvec' and args.relvec_initialize_mode == 'from_template':
            inputs['input_ids'] = self.convert_template_to_relvec(inputs['input_ids'], samples_list)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'], my_mode = self.f_mode, my_connector = self)
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1).cpu()
        
        token_ids_list = inputs['input_ids'].cpu().tolist()
        masked_indices_list = []
        for lis in token_ids_list:
            l_m = []
            for w_idx, w in enumerate(lis):
                if w == tokenizer.mask_token_id:
                    l_m.append(w_idx)
            assert(len(l_m) == 1)
            masked_indices_list.append(l_m)
        
        return log_probs, token_ids_list, masked_indices_list
    
    def get_optimizers(self, model, args):
        # copied from hugging face       
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        learning_rate = args.learning_rate
        param_mode = args.fewshot_ft_param_mode
        if args.fewshot_ft_param_mode_cur is not None:
            param_mode = args.fewshot_ft_param_mode_cur
        logger.info('in model.get_optimizers, current param_mode: %s', param_mode)

        if param_mode in ['all', 'fix_final_bias', 'fix_lm_head']:
            #assert(args.relation_mode == 'template')
            exclude_lis = []
            if param_mode == 'fix_final_bias':
                exclude_lis.append('lm_head.bias')
            if param_mode == 'fix_lm_head':
                exclude_lis.extend(['lm_head', 'roberta.embeddings.word_embeddings.weight'])
            if 'fix' in param_mode:
                logger.info('param_mode is %s, fixing the following params: %s', param_mode, str(exclude_lis))
            optimizer_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if (not any((nd in n) for nd in no_decay)) and (not any((nd in n) for nd in exclude_lis))],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any((nd in n) for nd in no_decay) and (not any((nd in n) for nd in exclude_lis))],
                    "weight_decay": 0.0,
                },
            ] 
            self.params_for_clip = optimizer_parameters[0]['params'] + optimizer_parameters[1]['params']
            if args.relation_mode == 'relvec':
                if args.fewshot_ft_param_all_lr_overwrite == False:
                    logger.info('note: using relvec_learning_rate for the relvec_params %f', args.relvec_learning_rate)
                    relvec_lr = args.relvec_learning_rate
                else:
                    logger.info('note: using overwrited learning_rate for the relvec_params %f', args.learning_rate) 
                    relvec_lr = args.learning_rate
                optimizer_parameters.append({'params': [self.relvec_params], 'weight_decay': args.relvec_weight_decay, 'lr': relvec_lr})
                #optimizer_parameters.append({'params': [self.relvec_params], 'weight_decay': args.relvec_weight_decay})
                if args.relvec_input_mode == 'mlp':
                    optimizer_parameters[-1]['params'].extend(list(self.relvec_mlp.parameters()))
                self.params_for_clip.extend(optimizer_parameters[-1]['params'])
     
            logger.info('number of all params: %d number of params that will actually optimize: %d', len(list(model.parameters())), len(self.params_for_clip))
        elif param_mode == 'only_final_bias':
            assert(args.relation_mode == 'template')
            optimizer_parameters = [{"weight_decay": 0.0, "params": [model.lm_head.decoder.bias]}]
            self.params_for_clip = [model.lm_head.decoder.bias]
        elif 'bitfit' in param_mode:
            if 'plusfhb' in param_mode:
                ps = [model.lm_head.decoder.bias]
            else:
                ps = []
            for i in range(len(model.roberta.encoder.layer)):
                ps.append(model.roberta.encoder.layer[i].attention.self.query.bias) 
                ps.append(model.roberta.encoder.layer[i].intermediate.dense.bias)
            optimizer_parameters = [{"weight_decay": 0.0, "params": list(ps)}]
            self.bitfit_params = list(ps)
            self.params_for_clip = list(ps)

            if args.relation_mode == 'relvec':
                if args.fewshot_ft_param_all_lr_overwrite == False:
                    logger.info('note: using relvec_learning_rate for the relvec_params %f', args.relvec_learning_rate)
                    relvec_lr = args.relvec_learning_rate
                else:
                    logger.info('note: using overwrited learning_rate for the relvec_params %f', args.learning_rate) 
                    relvec_lr = args.learning_rate
                optimizer_parameters.append({'params': [self.relvec_params], 'weight_decay': args.relvec_weight_decay, 'lr': relvec_lr})
                #optimizer_parameters.append({'params': [self.relvec_params], 'weight_decay': args.relvec_weight_decay})
                if args.relvec_input_mode == 'mlp':
                    optimizer_parameters[-1]['params'].extend(list(self.relvec_mlp.parameters()))
                self.params_for_clip.extend(optimizer_parameters[-1]['params'])

        elif param_mode == 'only_final_hidden_bias':
            assert(args.relation_mode == 'template')
            optimizer_parameters = [{"weight_decay": 0.0, "params": [model.lm_head.dense.bias]}]
            self.params_for_clip = [model.lm_head.dense.bias]
        elif param_mode == 'only_relvec':
            optimizer_parameters = [{'params': [self.relvec_params], 'weight_decay': args.relvec_weight_decay}]
            if args.relvec_input_mode == 'mlp':
                optimizer_parameters[0]['params'].extend(list(self.relvec_mlp.parameters()))
            self.params_for_clip = optimizer_parameters[0]['params']
            logger.info('only_relvec mode, using relvec_learning_rate %f', args.relvec_learning_rate)
            learning_rate = args.relvec_learning_rate

        optimizer = AdamW(optimizer_parameters, lr = learning_rate, eps = args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.warmup_steps, num_training_steps=args.max_training_steps)
        return optimizer, scheduler
    
    def convert_template_to_relvec(self, input_ids, samples):
        assert(len(input_ids) == len(samples))
        tokenizer, model, relvec_last_co = self.tokenizer, self.model, -1
        res_lis = []
        for k in range(len(input_ids)):
            sample = samples[k]
            ms_l, template_cur = input_ids[k].tolist(), sample['template']
            if not template_cur.startswith('[X]'):
                sub_l = tokenizer(' ' + sample['sub_label'])['input_ids'][1:-1]
            else:
                sub_l = tokenizer(sample['sub_label'])['input_ids'][1:-1]
            pos = [ms_l[i:i+len(sub_l)]==sub_l for i in range(len(ms_l)-len(sub_l)+1)]
            if sum(pos) != 1:
                print('sum(pos) != 1, something wrong!')
                breakpoint()
            sub_b, sub_e = pos.index(True), pos.index(True) + len(sub_l)
            relvec_idx = 0
            for i in range(len(ms_l)):
                if i >= sub_b and i < sub_e:
                    continue
                if ms_l[i] in [0, 2, tokenizer.mask_token_id, tokenizer.pad_token_id]:
                    continue
                ms_l[i] = self.relvec_idxs[relvec_idx]
                relvec_idx += 1
            if relvec_last_co != -1:
                assert(relvec_last_co == relvec_idx)
            relvec_last_co = relvec_idx
            res_lis.append(ms_l)
        res = torch.LongTensor(res_lis).cuda()
        assert(res.size() == input_ids.size())
        return res
    
    def truncate_512(self, sen_lis):
        tokenizer = self.tokenizer
        inputs = tokenizer(sen_lis, return_tensors="pt", padding = True)
        co = 0
        while inputs['input_ids'].size(1) > 512:
            if co == 0:
                logger.info('[[warning!!]] size larger than 512, doing front truncating...')
            co = co + 1
            for kk in range(6): #to make it faster, truncate the first 3 words
                sen_lis = [s[s.index(' ') + 1:] for s in sen_lis]
            inputs = tokenizer(sen_lis, return_tensors="pt", padding = True)
        return sen_lis

    def mlm_forward_step(self, mb, model, optimizer, scheduler, args, do_train = True):
        if do_train == True:
            model.train()
            optimizer.zero_grad() #in some cases, the thing to optimize contian self.relvec_params, which is not in model
        else:
            model.eval()
        tokenizer, bz, res_d = self.tokenizer, len(mb), {}

        sen_lis = [item['sentence'] for item in mb]
        targets = torch.LongTensor([item['obj_idx'] for item in mb]).cuda()
        
        sen_lis = self.truncate_512(sen_lis)
        inputs = tokenizer(sen_lis, return_tensors="pt", padding = True)

        assert(inputs['input_ids'].size(1) <= 512)
        for k in inputs: inputs[k] = inputs[k].cuda()
 
        if args.relation_mode == 'relvec' and args.relvec_initialize_mode == 'from_template':
            inputs['input_ids'] = self.convert_template_to_relvec(inputs['input_ids'], mb)

        token_ids_list = inputs['input_ids'].cpu().tolist()
        masked_indices_list = []
        for lis in token_ids_list:
            l_m = []
            for w_idx, w in enumerate(lis):
                if w == tokenizer.mask_token_id:
                    l_m.append(w_idx)
            assert(len(l_m) == 1)
            masked_indices_list.append(l_m[0])
        res_d['masked_indices_list'] = masked_indices_list
        
        outputs = model(**inputs, labels=inputs['input_ids'], my_mode = self.f_mode, my_connector = self)
        logits = outputs.logits
        logits = logits[list(range(logits.size(0))), masked_indices_list, :]
        loss = torch.nn.CrossEntropyLoss()
        ce_loss = loss(logits, targets)

        if do_train == True:
            #(ce_loss * bz).backward() #don't *ba here!!
            ce_loss.backward()
            if args.max_grad_norm > 0:
                #torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                total_norm = torch.nn.utils.clip_grad_norm_(self.params_for_clip, args.max_grad_norm)
                #print('debug total_norm:', total_norm)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                for i in self.relvec_idxs: #the relvec are stored and optimized elsewhere, so here I just set it to zero
                    model.roberta.embeddings.word_embeddings.weight[i] = 0

        return ce_loss.item(), res_d

    def analysis_final_bias(self, train_set):
        bias_cur, vocab = self.model.lm_head.decoder.bias, self.vocab
        s_lis = torch.argsort(self.model.lm_head.decoder.bias, descending = True).tolist() 
        logger.info('[analysis_final_bias]')
    
        logger.info('train_set obj_labels: %s', str([item['obj_label'] for item in train_set]))
        
        s_report = ''
        for i, idx in enumerate(s_lis[:30]):
            s_report += '{}_{} '.format(self.vocab[idx], str(bias_cur[idx].item())[:5])
        logger.info('sorted bias values after training: %s', s_report)

    def mlm_finetune(self, train_samples, dev_samples, seed, args, only_run_dev = False, do_log = True):
        def clean_data(samples):
            res = []
            for sample in samples:
                item = {}
                item = {'sentence': sample['masked_sentences'][0]}
                assert('[MASK]' in item['sentence'] or '<mask>' in item['sentence']) 
                item['sentence'] = item['sentence'].replace('[MASK]', '<mask>')
                assert((chr(288) + sample['obj_label']) in self.inverse_vocab_ori)
                item['obj_idx'] = self.inverse_vocab_ori[(chr(288) + sample['obj_label'])]
                item['sample'], item['obj_label'], item['sub_label'] = sample, sample['obj_label'], sample['sub_label']
                item['template'] = sample['template']
                res.append(item)
            return res
        train_set, dev_set = clean_data(train_samples), clean_data(dev_samples)
        if only_run_dev == False:
            optimizer, scheduler = self.get_optimizers(self.model, args)
        else:
            optimizer, scheduler = None, None

        if only_run_dev == True:
            dev_loss_lis = []
            for s in range(0, len(dev_set), 32):
                dev_loss_cur, res_d = self.mlm_forward_step(dev_set[s:s+32], self.model, optimizer, scheduler, args, do_train = False)
                dev_loss_lis.append(dev_loss_cur)
            return {'best_dev_loss': np.mean(dev_loss_lis)}
        
        model, best_dev_loss, state_best_dev, best_dev_step, relvec_best_dev, relvec_mlp_best_dev, res_d = self.model, 100, None, None, None, None, {}
        dev_interval = 10
        if 'bitfit' in args.fewshot_ft_param_mode:
            dev_interval = 5
            bitfit_params_ori = copy.deepcopy(self.bitfit_params)
            
        if do_log == True: 
            logger.info('dev_interval: %d', dev_interval)
        else:
            logger.info('do_log is False, logging will be skipped...')
        logger.info('mlm_finetune applying seed %d', seed)
        random.seed(seed) #for debugg!
        torch.manual_seed(seed)
        
        for step_idx in range(args.fewshot_ft_maxsteps + 1):
            mb_cur = []
            if args.fewshot_ft_batch_mode == 'random':
                for i in range(args.fewshot_ft_bz):
                    r_idx = random.randint(0, len(train_set) - 1)
                    mb_cur.append(train_set[r_idx])
            if args.fewshot_ft_batch_mode == 'full_batch':
                mb_cur = train_set

            train_loss_cur, res_d = self.mlm_forward_step(mb_cur, self.model, optimizer, scheduler, args, do_train = True if step_idx > 0 else False)
            
            if step_idx % dev_interval == 0:
                dev_loss_lis = []
                for s in range(0, len(dev_set), 32):
                    dev_loss_cur, res_d = self.mlm_forward_step(dev_set[s:s+32], self.model, optimizer, scheduler, args, do_train = False)
                    dev_loss_lis.append(dev_loss_cur)
                    #print('debug dev', s, len(dev_set[s:s+32]))
                dev_loss_cur = np.mean(dev_loss_lis)
                if do_log == True: 
                    logger.info('step: %d train_loss_cur: %f dev_loss_cur: %f', step_idx, train_loss_cur, dev_loss_cur)
                if dev_loss_cur < best_dev_loss:
                    best_dev_loss, best_dev_step = dev_loss_cur, step_idx
                    state_best_dev = copy.deepcopy(self.model.state_dict())
                    if self.relvec_params is not None:
                        relvec_best_dev = copy.deepcopy(self.relvec_params)
                    if self.relvec_mlp is not None:
                        relvec_mlp_best_dev = copy.deepcopy(self.relvec_mlp)
                    
                if dev_loss_cur > best_dev_loss * 1.4:
                    logger.info('dev_loss became worse, early stopped')
                    break
            #assert(best_dev_step < args.fewshot_ft_maxsteps - 1)
        
        self.model.load_state_dict(state_best_dev) #rolling back
        if self.relvec_params is not None:
            self.relvec_params = relvec_best_dev
        if self.relvec_mlp is not None:
            self.relvec_mlp = relvec_mlp_best_dev
        
        dev_loss_lis = []
        for s in range(0, len(dev_set), 32):
            dev_loss_cur, res_d = self.mlm_forward_step(dev_set[s:s+32], self.model, optimizer, scheduler, args, do_train = False)
            dev_loss_lis.append(dev_loss_cur)
        dev_loss_cur = np.mean(dev_loss_lis)
        logger.info('rolling back to state_best_dev, current dev_loss: %f recorded dev_loss: %f', dev_loss_cur, best_dev_loss)
        assert(abs(dev_loss_cur - best_dev_loss) < 0.01)
        #must be a method for train / eval of a single mini-batch
        #for training, sample randomly for each mb
        #for evaluation, for now assuming the whole dev set can be evaluated in one single mb

        if args.fewshot_ft_param_mode == 'only_final_bias':
            self.analysis_final_bias(train_set)               
        
        if 'bitfit' in args.fewshot_ft_param_mode:
            res_d['bitfit_params_ori'] = bitfit_params_ori
            res_d['bitfit_params_ft'] = copy.deepcopy(self.bitfit_params)

        res_d['best_dev_loss'], res_d['best_dev_step'] = best_dev_loss, best_dev_step
        return res_d

    def get_contextual_embeddings(self, sentences_list, try_cuda=True):
        return None
