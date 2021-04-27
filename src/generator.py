# author: Xiang Gao at Microsoft Research AI NLP Group

import torch, pdb
import numpy as np
from transformers19 import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import torch.nn.functional as F
from shared import EOS_token


class Generator:

    def __init__(self, opt):
        self.opt = opt
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model_config = GPT2Config(n_embd=1024, n_layer=24, n_head=16)        
        self.model = GPT2LMHeadModel(model_config)
        self.ix_EOS = 50256
        if self.opt.cuda:
            self.model.cuda()
        self.reset()
    

    def load(self, path=None):
        if path is None:
            path = self.opt.path_gen
            if self.opt.path_gen is None:
                return
        print('loading from '+path)
        weights = torch.load(path)
        if "lm_head.decoder.weight" in weights:
            weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
            weights.pop("lm_head.decoder.weight",None)
        self.model.load_state_dict(weights)

    
    def reset(self):
        self.load()
        if self.opt.T is None:
            TT = [0.3, 1, 3]
            self.T = np.random.choice(TT)
        else:
            self.T = self.opt.T


    def predict_sampling(self, cxt, n_hyp=5, return_partial=False, return_str=True, seed=2020):
        self.model.eval()
        cxt_seq = self.tokenize([cxt])
        tokens, _, _ = self.cat_ids(cxt_seq)
        tokens = tokens.repeat(n_hyp, 1)

        len_cxt = tokens.shape[-1]
        sum_logP = [0] * n_hyp
        live = [True] * n_hyp
        seqs = [[] for _ in range(n_hyp)]
        if seed is not None:
            np.random.seed(seed)
        past = None
        for t in range(self.opt.max_l_rsp):
            logits, past = self.model(tokens, past=past)
            prob = torch.softmax(logits[:, -1, :] / self.T, dim=-1)
            if self.opt.cuda:
                prob = prob.cpu()
            prob = prob.detach().numpy() + 1e-7 # avoid numerical errors
            vocab = prob.shape[-1]
            if t == 0:  # force each hyp starts with a different token
                next_tokens = np.random.choice(vocab, n_hyp, p=prob[0,:]/prob[0,:].sum(), replace=False)
            else:
                next_tokens = []
            for i in range(n_hyp):
                if t > 0:
                    next_token = np.random.choice(vocab, p=prob[i,:]/prob[0,:].sum())
                    next_tokens.append(next_token)
                else:
                    next_token = next_tokens[i]
                if not live[i]:
                    continue
                sum_logP[i] += np.log(prob[i, next_token])
                seqs[i].append(next_token)
                if next_token == self.ix_EOS:
                    live[i] = False
                    continue
            next_tokens = torch.LongTensor(next_tokens).view(-1, 1)
            if self.opt.cuda:
                next_tokens = next_tokens.cuda()
            #tokens = torch.cat([tokens, next_tokens], dim=-1)
            tokens = next_tokens    # don't cat as we use past

        ret = []
        for i in range(n_hyp):
            if live[i] and (not return_partial):
                continue
            prob = np.exp(sum_logP[i] / (len(seqs[i]) + 1))
            if return_str:
                hyp = self.tokenizer.decode(seqs[i]).strip('<|endoftext|>').strip()
            else:
                hyp = seqs[i]
            ret.append((prob, hyp))
        if return_str:
            return ret
        else:
            return cxt_seq[0], ret


    def loss(self, cxts, tgts):
        if isinstance(cxts[0], str):
            cxts, tgts = self.tokenize(cxts_str, tgts_str)
            add_hyp_EOS = True
        else:
            add_hyp_EOS = False
        ids, masks, labels = self.cat_ids(cxts, tgts, add_hyp_EOS=add_hyp_EOS)
        logits, _ = self.model(ids, attention_mask=masks)
        logP = F.log_softmax(logits, dim=-1)

        loss = F.nll_loss(
            logP.permute(0, 2, 1),  # [N, C, L]
            labels,                 # [N, L]
            ignore_index=-1,
            reduction='none',         # for RL re-weighting
            )
        avg = loss.sum(dim=-1) / (labels >= 0).sum(dim=-1).float()
        return avg

        
    def tokenize(self, cxts, rsps=None):
        """ list of str => list of ints """

        ids_cxt = []
        for cxt in cxts:
            _ids = []
            turns = cxt.split(EOS_token)
            for turn in turns:
                seq = self.tokenizer.encode(turn.strip())
                _ids += seq + [self.ix_EOS]
            ids_cxt.append(_ids[:-1])

        if rsps is None:
            return ids_cxt
            
        ids_rsps = []
        for rsp in rsps:
            ids_rsps.append(self.tokenizer.encode(rsp.strip()))

        return ids_cxt, ids_rsps


    def cat_ids(self, cxts, rsps=None, add_hyp_EOS=True):
        """ list of ints => tensors"""

        n = len(cxts)
        ids = []
        masks = []

        lens_inp = []
        lens_all = []
        for i in range(n):
            _ids_cxt = cxts[i][:]
            if len(_ids_cxt) > self.opt.max_l_cxt:
                _ids_cxt = _ids_cxt[-self.opt.max_l_cxt:]
            _ids_cxt.append(self.ix_EOS)

            if rsps is None:
                _ids_rsp = []
            else:
                _ids_rsp = rsps[i]
                if add_hyp_EOS:
                    _ids_rsp.append(self.ix_EOS)
                if len(_ids_rsp) > self.opt.max_l_rsp:
                    _ids_rsp = _ids_rsp[:self.opt.max_l_rsp]

            _ids = _ids_cxt + _ids_rsp
            l = len(_ids)
            lens_all.append(l)
            lens_inp.append(len(_ids_cxt))
            ids.append(_ids)

        # pad
        max_len = max(lens_all)
        labels = []
        for i in range(n):
            l = lens_all[i]
            l_inp = lens_inp[i]
            l_pad = max_len - l
            masks.append([1] * l + [0] * l_pad)
            labels.append([-1] * (l_inp - 1) + ids[i][l_inp:] + [-1] * (l_pad + 1))   # shifted left 1 time step
            ids[i] += [self.ix_EOS] * l_pad     # this should be done **after** labels

        ids = torch.LongTensor(ids)
        masks = torch.LongTensor(masks)
        labels = torch.LongTensor(labels)
        if self.opt.cuda:
            ids = ids.cuda()
            masks = masks.cuda()
            labels = labels.cuda()
            
        return ids, masks, labels


class OptionGenerator:
    def __init__(self, T=1):
        self.cuda = True
        self.path_gen = 'restore/medium_ft.pkl'
        self.max_l_cxt = 90
        self.max_l_rsp = 30
        self.T = T

