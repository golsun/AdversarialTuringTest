# author: Xiang Gao at Microsoft Research AI NLP Group


import torch, os, pdb, shutil
import numpy as np
from shared import EOS_token
from collections import Counter, defaultdict


class Feeder:

    def __init__(self, opt):
        self.opt = opt
        self.n_line = defaultdict(int)
        self.fileD = dict()
        self.fileG = dict()
        for sub in ['train', 'vali', 'test']:
            self.reset('G', sub)
            self.reset('D', sub)
        
        self.fldD = opt.fld_out + '/dataD'
        os.makedirs(self.fldD, exist_ok=True)
        if opt.prev_dataD is not None:
            for fname in os.listdir(opt.prev_dataD):
                if not fname.endswith('.tsv'):
                    continue
                print('copying '+fname)
                shutil.copy(opt.prev_dataD + '/' + fname, self.fldD + '/' + fname)
        n_child = 0
        while True:
            path = self.fldD + '/g%i_vali.tsv'%n_child
            if os.path.exists(path):
                for i, line in enumerate(open(path, encoding='utf-8')):
                    pass
                self.n_line[(n_child, 'vali')] = i
                _path = self.fldD + '/g%i_train.tsv'%n_child
                if os.path.exists(_path):
                    for i, line in enumerate(open(_path, encoding='utf-8')):
                        pass
                else:
                    i = 0
                self.n_line[(n_child, 'train')] = i
                n_child += 1
            else:
                #print('cannot find '+path)
                break
        self.n_child = n_child
        print('initial n_child = %i'%self.n_child)

        self.ix_EOS = 50256
        self.ix_OMT = 986
        self.dpt_EOS = ' EOS '

    
    def pathD(self, child, sub):
        return self.fldD + '/g%i_%s.tsv'%(child, sub)

    
    def birth(self):
        self.n_child += 1
        print('n_child updated to %i'%self.n_child)


    def reset(self, mode, sub):
        # mode is `G` or `D`
        path = '%s/%s.tsv'%(self.opt.fld_data[mode], sub)
        attr = self.fileG if mode == 'G' else self.fileD
        if os.path.exists(path):
            print('resetting %s-%s = %s'%(mode, sub, path))
            attr[sub] = open(path, encoding='utf-8', errors='ignore')
    

    def get_batchG(self, size, sub='train'):
        cxts = []
        rsps = []
        def clean_s(s):
            s = ' '.join(s.strip().split()[1:])
            for c in ',.?!':
                s = s.replace(' '+c, c)
            return s.strip()

        def read():
            for line in self.fileG[sub]:
                ss = line.strip('\n').split('\t')
                if len(ss) != 2:
                    continue
                cxt = (' ' + EOS_token + ' ').join([
                        clean_s(turn) for turn in ss[0].split(self.dpt_EOS)
                    ]).strip()
                rsp = clean_s(ss[1])
                if (not cxt) or (not rsp):
                    continue
                cxts.append(cxt)
                rsps.append(rsp)
                if len(cxts) == size:
                    break
                
        while True:
            read()
            if len(cxts) == size:
                break
            self.reset('G', sub)
        return cxts, rsps


    def reset_examples(self):
        # so we can only save vali examples from the latest generator
        child = self.n_child - 1
        sub = 'vali'
        open(self.pathD(child, sub), 'w')
        self.n_line[(child, sub)] = 0

    
    def save_samples(self, cxts, reals, fakes, sub='train'):
        """
        append the results to `cxt_real_fake` dataset, whose path is `self.pathD(child, sub)`
        """
        child = self.n_child - 1
        lines = []
        for i in range(len(cxts)):
            line = '\t'.join([cxts[i], reals[i], fakes[i]])
            lines.append(line)
        with open(self.pathD(child, sub), 'a', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        self.n_line[(child, sub)] += len(lines)


    def get_batchD_child(self, size, child, sub='train'):
        """
        randomly sample from dataset generated using child-generators
        """
        cxts = []
        reals = []
        fakes = []
        picked = np.random.choice(self.n_line[(child, sub)], size, replace=False)
        picked = set(picked.tolist())
        if self.opt.verbose:
            print(picked)
        for i, line in enumerate(open(self.pathD(child, sub), encoding='utf-8')):
            if i in picked:
                ss = line.strip('\n').split('\t')
                if len(ss) != 3:
                    continue
                cxts.append(ss[0].strip())
                reals.append(ss[1].strip())
                fakes.append(ss[2].strip())

        if self.opt.debug:
            pdb.set_trace()
        return cxts, reals, fakes

        
    def get_batchD_parent(self, size, sub='train'):
        """
        continue to read samples from parent dataset
        """
        cxts = []
        reals = []
        fakes = []

        def read():
            for line in self.fileD[sub]:
                ss = line.strip('\n').split('\t')
                if len(ss) < 3:
                    continue
                cxt = ss[0].strip()
                real = ss[1].strip()
                fake = ss[2].strip()
                if (not cxt) or (not real) or (not fake):
                    continue
                cxts.append(cxt)
                reals.append(real)
                fakes.append(fake)
                if len(cxts) == size:
                    break
                
        while True:
            read()
            if len(cxts) == size:
                break
            self.reset('D', sub)
        return cxts, reals, fakes

    
    def get_batchD(self, size, sub='train', mix=None, p=None):
        if self.n_child == 0:
            return self.get_batchD_parent(size, sub)
        
        if mix is None:
            cand = ['parent'] + list(range(self.n_child))
            xx = np.random.choice(cand, size, p=p)
            mix = Counter(xx)
        cxts = []
        reals = []
        fakes = []
        if self.opt.verbose:
            print(mix)
        for k in mix:
            if k == 'parent':
                _cxts, _reals, _fakes = self.get_batchD_parent(mix[k], sub)
            else:
                _cxts, _reals, _fakes = self.get_batchD_child(mix[k], int(k), sub)
            cxts += _cxts
            reals += _reals
            fakes += _fakes

        # shuffle
        ii = list(range(len(cxts)))
        np.random.shuffle(ii)
        cxts = [cxts[i] for i in ii]
        reals = [reals[i] for i in ii]
        fakes = [fakes[i] for i in ii]

        return cxts, reals, fakes