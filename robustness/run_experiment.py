import os, itertools, random, numpy as np, pandas as pd
from _common import load_model, load_test, daily_instruments
from attn_patch import install_patch
from paired_eval import paired_eval

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_HERE, 'results')


def all_instruments(dl_test):
    return sorted({s for _, insts in daily_instruments(dl_test) for s in insts})


def sample_corrupt_set(universe_list, frac, seed):
    rng = random.Random(seed)
    k = int(round(frac * len(universe_list)))
    return set(rng.sample(universe_list, k))


def main(universe='csi300', prefix='opensource'):
    os.makedirs(_RESULTS, exist_ok=True)
    dl_test = load_test(universe, prefix)
    insts = all_instruments(dl_test)

    ks      = [0.0, 0.05, 0.10, 0.20, 0.40]
    types   = ['gaussian', 'missing']  # shock 제외
    sigma   = 2.0
    seeds   = [0, 1, 2, 3, 4]
    conds   = [False, True]           # self_only: ON=False, OFF=True

    rows = []
    for self_only in conds:
        model = load_model(universe, prefix, seed=0)   # 가중치는 고정(seed0 ckpt)
        sattn = install_patch(model)
        p0_cache = {}                                  # 조건별 캐시
        for ctype, k, seed in itertools.product(types, ks, seeds):
            cset = set() if k == 0 else sample_corrupt_set(insts, k, seed)
            res, p0_cache = paired_eval(model, sattn, dl_test, cset, ctype,
                                        sigma, seed, self_only, p0_cache)
            rows.append(dict(universe=universe, self_only=self_only,
                             ctype=ctype, k=k, sigma=sigma, seed=seed, **res))
            print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(_RESULTS, f'main_{universe}.csv'), index=False)


if __name__ == '__main__':
    main('csi300', 'opensource')
