# author: Xiang Gao at Microsoft Research AI NLP Group


import torch, os, pdb
import numpy as np
from transformers19 import GPT2Tokenizer, GPT2Model, GPT2Config
from shared import EOS_token


class OptionInfer:
    def __init__(self, cuda=True):
        self.cuda = cuda
        self.max_l_cxt = 60
        self.max_l_rsp = 30


class ScorerBase(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.ix_EOS = 50256
        self.ix_OMT = 986
        self.opt = opt
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    
    def predict(self, cxt, hyps, max_cxt_turn=None, return_logits=False):
        """
        return scores of hyps given cxt
        # cxt = str
        # hyps = list of str
        # scores = dict, keys = ['score', ...], val = np array
        """

        self.eval()
        ids_cxt, ids_rsps_pos = self.tokenize([cxt] * len(hyps), hyps)
        batch = self.cat_ids(ids_cxt, ids_rsps_pos)
        with torch.no_grad():
            scores = self.core(batch['ids_pos'], batch['lens_pos'], return_logits=return_logits)
    
        if self.opt.cuda:
            scores = scores.cpu()
        return scores.detach().numpy()
    

    def play(self):
        while True:
            cxt = input('\ncxt:\t')
            if not cxt:
                break
            while True:
                hyp = input('hyp:\t')
                if not hyp:
                    break
                score = self.predict(cxt, [hyp])
                print('%.3f'%score[0])


    def tokenize(self, cxts, rsps_pos, rsps_neg=None):
        """ list of str => list of ints """

        ids_cxt = []
        for cxt in cxts:
            _ids = []
            turns = cxt.split(EOS_token)
            for turn in turns:
                seq = self.tokenizer.encode(turn.strip())
                _ids += seq + [self.ix_EOS]
            ids_cxt.append(_ids[:-1])

        ids_rsps_pos = []
        for rsp in rsps_pos:
            ids_rsps_pos.append(self.tokenizer.encode(rsp.strip()))

        if rsps_neg is None:
            return ids_cxt, ids_rsps_pos
            
        ids_rsps_neg = []
        for rsp in rsps_neg:
            ids_rsps_neg.append(self.tokenizer.encode(rsp.strip()))

        return ids_cxt, ids_rsps_pos, ids_rsps_neg


    def cat_ids(self, cxts, rsps_pos, rsps_neg=None):
        """ list of ints => tensors"""

        n = len(cxts)
        ids_pos = []
        ids_neg = []
        lens_pos = []
        lens_neg = []

        for i in range(n):
            _ids_cxt = cxts[i][:]
            if len(_ids_cxt) > self.opt.max_l_cxt:
                _ids_cxt = _ids_cxt[-self.opt.max_l_cxt:]
            _ids_cxt.append(self.ix_EOS)

            _ids_rsp_pos = rsps_pos[i] + [self.ix_EOS]
            if len(_ids_rsp_pos) > self.opt.max_l_rsp:
                _ids_rsp_pos = _ids_rsp_pos[:self.opt.max_l_rsp]

            _ids_pos = _ids_cxt + _ids_rsp_pos
            ids_pos.append(_ids_pos)
            lens_pos.append(len(_ids_pos))

            if rsps_neg is not None:
                _ids_rsp_neg = rsps_neg[i] + [self.ix_EOS]
                if len(_ids_rsp_neg) > self.opt.max_l_rsp:
                    _ids_rsp_neg = _ids_rsp_neg[:self.opt.max_l_rsp]

                _ids_neg = _ids_cxt + _ids_rsp_neg
                ids_neg.append(_ids_neg)
                lens_neg.append(len(_ids_neg))

        max_len = max(lens_pos)
        for i in range(n):
            l_pad = max_len - lens_pos[i]
            ids_pos[i] += [self.ix_EOS] * l_pad
        ids_pos = torch.LongTensor(ids_pos)
        if self.opt.cuda:
            ids_pos = ids_pos.cuda()
        ret = {
            'ids_pos': ids_pos,
            'lens_pos': lens_pos,
            }
        
        if rsps_neg is not None:
            max_len = max(lens_neg)
            for i in range(n):
                l_pad = max_len - lens_neg[i]
                ids_neg[i] += [self.ix_EOS] * l_pad
            ids_neg = torch.LongTensor(ids_neg)
            if self.opt.cuda:
                ids_neg = ids_neg.cuda()
            ret['ids_neg'] = ids_neg
            ret['lens_neg'] = lens_neg

        return ret


    def forward(self, cxts, rsps_pos, rsps_neg):
        ids_cxt, ids_rsps_pos, ids_rsps_neg = self.tokenize(cxts, rsps_pos, rsps_neg)
        batch = self.cat_ids(ids_cxt, ids_rsps_pos, ids_rsps_neg)
        logits_pos = self.core(batch['ids_pos'], batch['lens_pos'], return_logits=True)
        logits_neg = self.core(batch['ids_neg'], batch['lens_neg'], return_logits=True)
        # softmax to get the `probability` to rank pos/neg correctly
        probs = torch.exp(logits_pos) / (torch.exp(logits_pos) + torch.exp(logits_neg))
        L = - torch.log(probs).mean()
        return L, probs



class Scorer(ScorerBase):
    def __init__(self, opt):
        super().__init__(opt)
        n_embd = 1024
        config = GPT2Config(n_embd=n_embd, n_layer=24, n_head=16)
        self.transformer = GPT2Model(config)
        self.score = torch.nn.Linear(n_embd, 1, bias=False)
        

    def core(self, ids, l_ids, return_logits=False, T=1):
        n = ids.shape[0]
        attention_mask = torch.ones_like(ids)
        for i in range(n):
            attention_mask[i, l_ids[i]:] *= 0
        hidden_states, _ = self.transformer(ids, attention_mask=attention_mask)
        logits = self.score(hidden_states).squeeze(-1)
        logits = torch.stack([logits[i, l_ids[i] - 1] for i in range(n)])
        if return_logits:
            return logits
        else:
            return torch.sigmoid(logits/T)

    
    def load(self, path):
        if os.getenv('PT_DATA_DIR'):
            path = path.replace('PT_DATA', os.getenv('PT_DATA_DIR'))
        print('loading from '+path)
        weights = torch.load(path)
        if path.endswith('.pkl'):
            # DialoGPT checkpoint
            weights['score.weight'] = weights['lm_head.decoder.weight'][self.ix_EOS: self.ix_EOS+1, :]
            del weights['lm_head.decoder.weight']
        self.load_state_dict(weights)



class JointScorer(ScorerBase):
    
    def core(self, ids, l_ids, return_logits=False, T=1):
        sum_logits_wt = 0
        sum_wt = 0
        vv = []
        for k in self.kk:
            scorer = getattr(self, 'scorer_%s'%k)
            if self.trainable[k]:
                logits = scorer.core(ids, l_ids, return_logits=True)
            else:
                with torch.no_grad():
                    logits = scorer.core(ids, l_ids)
            vv.append(logits.unsqueeze(-1))
        
        vv = torch.cat(vv, dim=-1)          # [batch, kk]
        ww = torch.softmax(-vv, dim=-1)     # [batch, kk]
        logits = (ww * vv).sum(dim=-1)      # [batch]
        if return_logits:
            return logits
        else:
            return torch.sigmoid(logits/T)
        
    
    def load(self, path_config):
        import yaml
        with open(path_config, 'r') as stream:
            config = yaml.safe_load(stream)
        print(config)

        paths = dict()
        self.kk = []
        self.wt = dict()
        self.trainable = dict()
        for d in config['member']:
            k = d['name']
            self.kk.append(k)
            self.wt[k] = d['wt']
            self.trainable[k] = d['trainable']
            paths[k] = d['path']
        
        for k in paths:
            print('setting up model `%s`'%k)
            scorer = Scorer(OptionInfer(cuda=self.opt.cuda))
            scorer.load(paths[k])
            if self.opt.cuda:
                scorer.cuda()
            setattr(self, 'scorer_%s'%k, scorer)


def load_scorer(opt):
    if opt.path_scorer.endswith('yml') or  opt.path_scorer.endswith('yaml'):
        scorer = JointScorer(opt)
    else:
        scorer = Scorer(opt)
    scorer.load(opt.path_scorer)
    if opt.cuda:
        scorer = scorer.cuda()
    return scorer