# author: Xiang Gao at Microsoft Research AI NLP Group


import argparse, torch, time, pdb, sys
import os, socket
from master import MasterGAN


class Option:

    def __init__(self, args):        
        if args.cpu or not torch.cuda.is_available():
            self.cuda = False 
        else:
            self.cuda = True
        self.task = args.task
        self.batch = {'G':args.batchG, 'D':args.batchD} 
        self.vali_size = args.vali_size
        self.vali_print = args.vali_print
        self.lr = {'G': args.lrG, 'D': args.lrD}
        self.max_l_cxt = args.max_l_cxt
        self.max_l_rsp = args.max_l_rsp
        self.max_n_hyp = args.max_n_hyp
        self.wt_rl = args.wt_rl
        self.debug = args.debug
        self.switch = {'D': args.switchD, 'G': args.switchG}
        self.acc_switch = {'G':args.accG, 'D':args.accD}
        self.csize = args.csize
        self.T = args.T
        self.verbose = args.verbose
        self.hostname = socket.gethostname()
        self.prev_dataD = args.prev_dataD
        self.last = args.last

        if os.getenv('PT_DATA_DIR') is not None:
            self.fld_data = {
                'G': os.getenv('PT_DATA_DIR') + '/data/' + args.dataG,
                'D': os.getenv('PT_DATA_DIR') + '/data/' + args.dataD,
                } 
            self.fld_out = os.getenv('PT_OUTPUT_DIR') + '/out/%i'%time.time()
            self.path_gen = args.path_gen.replace('PT_DATA', os.getenv('PT_DATA_DIR'))
            self.path_scorer = args.path_scorer.replace('PT_DATA', os.getenv('PT_DATA_DIR'))
        else:
            self.fld_data = {
                'G': args.dataG,
                'D': args.dataD,
                }
            self.path_gen = args.path_gen
            self.path_scorer = args.path_scorer
            self.fld_out = 'out/%i'%time.time()

        os.makedirs(self.fld_out, exist_ok=True)

        self.topk = 3
        self.topp = 0.8
        self.beam = 10
        self.clip = 1
        self.step_max = 1e6
        self.step_print = 10
        self.step_vali = args.step_vali
        self.step_save = 50 if self.debug else 500

        print('@'*20)
        print('HOST  %s'%self.hostname)
        print('DATA  %s'%self.fld_data)
        print('OUT   %s'%self.fld_out)
        print('BATCH %s/%i'%(self.batch, torch.cuda.device_count()))
        print('@'*20)
        sys.stdout.flush()

    def save(self):
        d = self.__dict__
        lines = []
        for k in d:
            lines.append('%s\t%s'%(k, d[k]))
        with open(self.fld_out + '/opt.tsv', 'w') as f:
            f.write('\n'.join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('--dataG', type=str)
    parser.add_argument('--dataD', type=str)
    parser.add_argument('--batchG', type=int, default=32)
    parser.add_argument('--batchD', type=int, default=32)
    parser.add_argument('--vali_size', type=int, default=128)
    parser.add_argument('--vali_print', type=int, default=3)
    parser.add_argument('--lrG', type=float, default=3e-5)
    parser.add_argument('--lrD', type=float, default=1e-4)
    parser.add_argument('--path_gen','-pg', type=str)
    parser.add_argument('--path_scorer','-ps', type=str)
    parser.add_argument('--path_test', type=str)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--max_l_cxt', type=int, default=60)
    parser.add_argument('--max_l_rsp', type=int, default=30)
    parser.add_argument('--max_n_hyp', type=int, default=10)
    parser.add_argument('--wt_rl', type=float, default=1.)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--switchG', type=int, default=3000)
    parser.add_argument('--switchD', type=int, default=1000)
    parser.add_argument('--csize', type=int, default=1000)
    parser.add_argument('--accG', type=float, default=0.5, help="stop training G if min(acc) <= accG")
    parser.add_argument('--accD', type=float, default=0.7, help="stop training D if min(acc) >= accD")
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--step_vali', type=int, default=100)
    parser.add_argument('--prev_dataD', type=str)
    parser.add_argument('--T', type=float)
    parser.add_argument('--last', action='store_true')
    args = parser.parse_args()

    opt = Option(args)
    if args.task == 'play':
        from scorer import load_scorer
        scorer = load_scorer(opt)
        scorer.play()
    
    else:
        master = MasterGAN(opt)
        if args.task == 'train':
            master.train()
        elif args.task == 'vali':
            master.vali()
        elif args.task == 'test':
            master.test(args.path_test)