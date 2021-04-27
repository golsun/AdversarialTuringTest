# author: Xiang Gao at Microsoft Research AI NLP Group

import torch, pdb, os, time
import numpy as np
from generator import Generator, OptionGenerator


def attack(generator, path_src, n_hyp=5, n_src=None, batch=32, T=1):
    path_out = path_src + '.%ihyps_T%.1f.tsv'%(n_hyp, T)
    if os.path.exists(path_out):
        print('already: '+path_out)
        return

    assert(n_hyp % batch == 0)
    n_batch = int(n_hyp / batch)
    t0 = time.time()
    for i, line in enumerate(open(path_src, encoding='utf-8')):
        if i == n_src:
            break
        src = line.strip('\n')
        lines = []
        for j in range(n_batch):
            with torch.no_grad():
                ret = generator.predict_sampling(src, n_hyp=batch, return_partial=True, T=T, seed=None)
            for prob, hyp in ret:
                lines.append('\t'.join([src, '%.4f'%prob, hyp]))
            speed = (i + (j + 1)/n_batch) / (time.time() - t0) * 3600
            print('src %i/%s batch %i/%i, speed %.1f src/hr'%(i, n_src, j, n_batch, speed))
        with open(path_out, 'a', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')


if __name__ == "__main__":
    """
     python src/generation.py --path=data/src_only.txt --n_hyp=1024 --T=100 --n_src=1000
    """
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--n_hyp', type=int, default=5)
    parser.add_argument('--n_src', type=int)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--T', type=float, default=1.)
    args = parser.parse_args()

    cuda = False if args.cpu else torch.cuda.is_available()
    opt = OptionGenerator()
    generator = Generator(opt)
    attack(generator, args.path, n_hyp=args.n_hyp, n_src=args.n_src, batch=args.batch)