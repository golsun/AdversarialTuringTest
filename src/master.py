# author: Xiang Gao at Microsoft Research AI NLP Group


import torch, os, pdb, time, sys, warnings, datetime
import numpy as np
from feeder import Feeder
from generator import Generator
from scorer import load_scorer


class MasterGAN:
    """
    RL(+SL) to train generator G. G outputs fake examples to train D (together with all "old" data).
    SL to train classifier D.
    trained alternatingly.
    """

    def __init__(self, opt):
        self.opt =  opt
        self.generator = Generator(opt)
        self.scorer = load_scorer(opt)
        self.parallel()
        self.feeder = Feeder(opt)
        self.penalty = 1.   # for unfinished or empty response
        if opt.task == 'train':
            opt.save()
            os.makedirs(opt.fld_out + '/ckpt', exist_ok=True)
            self.path_log = self.opt.fld_out + '/log.txt'
        else:
            self.path_log = self.opt.fld_out + '/log_infer.txt'
    
    
    def print(self, s=''):
        try:
            print(s)
        except UnicodeEncodeError:
            print('[UnicodeEncodeError]')
            pass
        with open(self.path_log, 'a', encoding='utf-8') as f:
            f.write(s+'\n')


    def parallel(self):
        if self.opt.cuda:
            self.generator.model = self.generator.model.cuda()

        n_gpu = torch.cuda.device_count()
        if self.opt.cuda and n_gpu > 1:
            print('paralleling on %i GPU'%n_gpu)
            self.generator.model = torch.nn.DataParallel(self.generator.model)
            self.scorer = torch.nn.DataParallel(self.scorer)
            # after DataParallel, a warning about RNN weights shows up every batch
            warnings.filterwarnings("ignore")
            # after DataParallel, attr of self.model become attr of self.model.module
            self._modelG = self.generator.model.module
            self._modelD = self.scorer.module
        else:
            self._modelG = self.generator.model
            self._modelD = self.scorer
        if self.opt.task == 'train':
            self.optimizer = {
                'G': torch.optim.Adam(self._modelG.parameters(), lr=self.opt.lr['G']),
                'D': torch.optim.Adam(self._modelD.parameters(), lr=self.opt.lr['D']),
                }


    def rl_loss(self, cxts, debug=False):
        """
        for RL, we want to maximize expected reward
                argmax: E_s[r(a|s)]
        where r is reward of action a given state s
        based on REINFORCE policy gradient
                ∇ E_s[r(a|s)] = E_s[r(a|s) * ∇ logP(a|s)]
        so, to maximize E[r], we minimize Loss 
                argmin: Loss = E_s[- logP(a|s) * r(a|s)]
        in our case:
                Loss = E[NLL(hyp|cxt) * r(hyp|cxt)]
        """
        
        results = []
        L = 0
        np.random.seed(2020)
        for i, cxt in enumerate(cxts):
            result = []
            with torch.no_grad():
                cxt_seq, prob_seq = self.generator.predict_sampling(cxt, return_partial=True, return_str=False)
                probs = [prob for prob, _ in prob_seq]
                hyp_seqs = [seq for _, seq in prob_seq]
                hyps = [self.generator.tokenizer.decode(seq).strip('<|endoftext|>') for seq in hyp_seqs]
                logits = self.scorer.predict(cxt, hyps, return_logits=True)

                # reward
                unfinished = np.array([float(seq[-1] != self.generator.ix_EOS) for seq in hyp_seqs])
                empty = np.array([float(len(hyp.strip()) == 0) for hyp in hyps])
                reward = logits - self.penalty * (empty + unfinished)
                reward = (reward - reward.mean()) #/ (reward.std() + 1e-7)
                result = {'cxt':cxt, 'hyps':hyps, 'probs':probs, 'scores':logits, 'rewards':reward}
                reward = torch.FloatTensor(reward)
                if self.opt.cuda:
                    reward = reward.cuda()
                
            self.generator.model.train()
            cxt_seqs = [cxt_seq] * len(hyps)
            nll = self.generator.loss(cxt_seqs, hyp_seqs)
            L = L + (nll * reward).mean()
            
            if self.opt.cuda:
                nll_np = nll.cpu()
            nll_np = nll_np.detach().numpy()
            result['probs_nll'] = np.exp(-nll_np)
            results.append(result)

        if debug:
            pdb.set_trace()
        L = L / len(cxts)
        return L, results


    def save_samples(self, results, tgts, sub='train'):
        """
        results is list of dict with keys 'cxt', 'hyps', 'probs', 'scores', 'rewards'
        """
        cxts = []
        reals = []
        fakes = []
        for i in range(len(tgts)):
            cxt = results[i]['cxt']
            real = tgts[i]
            for fake in results[i]['hyps']:
                cxts.append(cxt)
                reals.append(real)
                fakes.append(fake)
        self.feeder.save_samples(cxts, reals, fakes, sub=sub)


    def train(self):
        def p_by_acc(acc):
            #logits = np.exp(1 - np.array(acc))
            #return logits / sum(logits)
            err = 1 - np.array(acc)
            return err/err.sum()

        def str_p(p):
            return '[' + ', '.join(['%.3f'%x for x in p]) + ']'

        MODE = 'G'
        self.feeder.birth()
        vali_loss, vali_perf, vali_corpora = self.vali()
        p_corpora = p_by_acc(vali_corpora)

        step = {'G':0, 'D':0}
        self.print('initial p_corpora = ' + str_p(p_corpora))
        self.print('initial generator T = %s'%self.generator.T)
        global_step = 0
        step_switch = self.opt.switch['G']
        t0 = time.time()
        TURN = 0
        while global_step < self.opt.step_max:
            other_mode = 'G' if MODE == 'D' else 'D'
            self.optimizer[MODE].zero_grad()

            if MODE == 'G':
                self.generator.model.train()
                cxts, tgts = self.feeder.get_batchG(self.opt.batch['G'])
                loss, results = self.rl_loss(cxts)
                self.save_samples(results, tgts)
            elif MODE == 'D':
                self.scorer.train()
                cxts, reals, fakes = self.feeder.get_batchD(self.opt.batch['D'], p=p_corpora)
                loss, _ = self.scorer.forward(cxts, reals, fakes)
            
            loss = loss.mean()
            loss.backward()
            self.optimizer[MODE].step()
            step[MODE] += 1
            global_step += 1
            info = '%s step %i (TURN %i G %i D %i)'%(MODE, global_step, TURN, step['G'], step['D'])

            if global_step % self.opt.step_print == 0:
                speed = (global_step / 1e3) / ((time.time() - t0) / 3600)
                self.print('%s speed %.2f loss %.4f'%(
                    info,
                    speed, 
                    loss,
                    ))

            switch = global_step == step_switch
            if switch:
                self.print('will switch from %s as step %i == step_switch'%(MODE, global_step))
                #if MODE == 'G':
                #    self.save(self.opt.fld_out + '/ckpt/final.pth')
                #    exit()
            if global_step % self.opt.step_vali == 0:
                vali_loss, vali_perf, vali_corpora = self.vali(info)
                if self.opt.last:
                    p_corpora = [0.] * (len(vali_corpora) - 1) + [1.]
                else:
                    p_corpora = p_by_acc(vali_corpora)
                self.print('updated p_corpora = ' + str_p(p_corpora))
                sys.stdout.flush()
                acc = vali_perf['D']
                if MODE == 'G' and acc < self.opt.acc_switch['G']:
                    self.print('G>D switch as %.3f < %.3f'%(acc, self.opt.acc_switch['G']))
                    switch = True
                if MODE == 'D' and acc > self.opt.acc_switch['D']:
                    self.print('D>G switch as %.3f > %.3f'%(acc, self.opt.acc_switch['D']))
                    switch = True

            if global_step % self.opt.step_save == 0:
                self.save(self.opt.fld_out + '/ckpt/turn%i.pth'%(int(TURN/10)*10))

            if switch:
                if MODE == 'G':
                    cbatch = int(self.opt.csize / self.opt.batch['G']) - step['G']
                    for i in range(cbatch):
                        if i % 10 == 0:
                            print('generating adv %i/%i'%(i, cbatch))
                        self.generator.model.eval()
                        cxts, tgts = self.feeder.get_batchG(self.opt.batch['G'])
                        with torch.no_grad():
                            loss, results = self.rl_loss(cxts)
                        self.save_samples(results, tgts)

                MODE = other_mode
                if MODE == 'G':
                    TURN += 1
                    step = {'G':0, 'D':0}
                step_switch = global_step + self.opt.switch[MODE]
                if self.opt.verbose:
                    self.print('MODE switched to %s, until %i'%(MODE, step_switch))
                if MODE == 'G':
                    if not self.opt.last:
                        self.generator.reset()
                    self.feeder.birth()
                    self.print('generator reset T = %s'%self.generator.T)


    def vali(self, info=''):
        d_loss = {}
        d_perf = {}

        # vali G -------------------------------------------------
        # do this first to get samples for D
        self.generator.model.eval()
        sum_loss = 0
        sum_score = 0
        n_batch = max(1, int(self.opt.vali_size/self.opt.batch['G']))
        self.feeder.reset('G', 'vali')
        n_print = 0
        self.feeder.reset_examples()
        for _ in range(n_batch):
            cxts, tgts = self.feeder.get_batchG(self.opt.batch['G'], sub='vali')
            with torch.no_grad():
                loss, results = self.rl_loss(cxts, debug=self.opt.debug)
                if self.opt.task == 'train':
                    self.save_samples(results, tgts, sub='vali')
                sum_loss += loss.mean().item()
                sum_score += np.mean([d['scores'].mean() for d in results])

            for i, d in enumerate(results):
                if n_print == self.opt.vali_print:
                    break
                self.print('\ncxt:\t%s'%d['cxt'])
                tgt = tgts[i]
                score = self.scorer.predict(d['cxt'], [tgt], return_logits=True)
                ss = ['tgt:', 'score %.2f'%score[0], ' '*11, tgt]
                self.print('\t'.join(ss))
                for j in range(len(d['hyps'])):
                    ss = ['hyp:', 'score %.2f'%d['scores'][j], 'reward %.2f'%d['rewards'][j], d['hyps'][j]]
                    self.print('\t'.join(ss))
                n_print += 1
        
        d_loss['G'] = sum_loss / n_batch
        d_perf['G'] = sum_score / n_batch
        
        s = info + 'lossG %.4f scoreG %.3f'%(d_loss['G'], d_perf['G'])
        self.print('[vali] ' + s.strip())

        # vali D -------------------------------------------------
        self.scorer.eval()
        corpora = ['parent'] + list(range(self.feeder.n_child))
        n_batch = max(1, int(self.opt.vali_size/self.opt.batch['D']))
        avg_loss = []
        avg_acc = []
        for corpus in corpora:
            if corpus == 'parent':
                self.feeder.reset('D', 'vali')
            sum_loss = 0
            sum_acc = 0
            for _ in range(n_batch):
                cxts, reals, fakes = self.feeder.get_batchD(self.opt.batch['D'], sub='vali', mix={corpus:self.opt.batch['D']})
                with torch.no_grad():
                    loss, accs = self.scorer.forward(cxts, reals, fakes)
                    sum_loss += loss.mean().item()
                    sum_acc += accs.mean().item()
            _loss = sum_loss / n_batch
            _acc = sum_acc / n_batch
            self.print('    - %s loss %.4f acc %.3f'%(corpus, _loss, _acc))
            avg_loss.append(_loss)
            avg_acc.append(_acc)
        #print('loss mean %.2f min %.2f max %.2f'%(np.mean(avg_loss), np.min(avg_loss), np.max(avg_loss)))
        print('acc mean %.2f min %.2f max %.2f'%(np.mean(avg_acc), np.min(avg_acc), np.max(avg_acc)))
        if self.opt.last:
            d_loss['D'] = avg_acc[-1]
            d_perf['D'] = avg_acc[-1]
        else:
            d_loss['D'] = np.mean(avg_loss)
            d_perf['D'] = np.min(avg_acc)
        return d_loss, d_perf, avg_acc


    def save(self, path):
        torch.save(self._modelG.state_dict(), path + '.G')
        if hasattr(self._modelD, 'trainable'):
            for k in self._modelD.trainable:
                if self._modelD.trainable[k]:
                    print('saving '+k)
                    model = getattr(self._modelD, 'scorer_%s'%k)
                    torch.save(model.state_dict(), path + '.%s'%k)
        else:
            torch.save(self._modelD.state_dict(), path + '.D')
        self.print('saved to '+path)


    def test(self, path, max_n=1024):
        self.scorer.eval()

        if os.path.isdir(path):
            paths = []
            for fname in os.listdir(path):
                if 'train' not in fname and fname.endswith('.tsv'):
                    paths.append(path + '/' + fname)
            paths = sorted(paths)
        else:
            paths = [path]
        
        path_out = path + '.eval(%s).txt'%self.opt.path_scorer.split('/')[-1].split('.')[0]
        with open(path_out, 'a', encoding='utf-8') as f:
            f.write('\n\n' + '===== %s ===='%datetime.datetime.now() + '\n\n')

        n_path = len(paths)
        for i, path in enumerate(paths):
            print('testing %i/%i: '%(i, n_path) + path)
            cxts = []
            reals = []
            fakes = []
            sum_acc = 0
            sum_n = 0
            for line in open(path, encoding='utf-8'):
                ss = line.strip('\n').split('\t')
                if len(ss) < 3:
                    continue
                cxts.append(ss[0].strip())
                reals.append(ss[1].strip())
                fakes.append(ss[2].strip())
                if len(cxts) == self.opt.batch['D']:
                    with torch.no_grad():
                        _, accs = self.scorer.forward(cxts, reals, fakes)
                        sum_acc += accs.sum().item()
                        sum_n += len(cxts)
                    cxts = []
                    reals = []
                    fakes = []
                    print('%i: %.4f'%(sum_n, sum_acc/sum_n))
                    if sum_n >= max_n:
                        break
                    
            if cxts:
                with torch.no_grad():
                    _, accs = self.scorer.forward(cxts, reals, fakes)
                    sum_acc += accs.mean().item()
                    sum_n += len(cxts)
            
            acc = sum_acc/sum_n if sum_n > 0 else np.nan
            s = '%.4f\t%i\t%s'%(acc, sum_n, path)
            print(s)
            with open(path_out, 'a', encoding='utf-8') as f:
                f.write(s + '\n')
                
        print('eval saved in: %s'%path_out)