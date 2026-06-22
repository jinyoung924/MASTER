"""C3 — 허브 오염 vs 무작위 오염 비교.

같은 오염 비율 k에서 corrupt_set을
  (1) top_hubs(centrality, k)  — SAttention에서 가장 주목을 많이 받는 종목,
  (2) 무작위 추출
로 바꿔 paired_eval을 돌리고 clean_IC / dP를 비교한다.

허브 가설: 허브를 오염시키면 무작위 종목을 오염시킬 때보다 깨끗한 종목으로
오염이 더 크게 전파된다(dP↑, clean_IC 저하↑). 종목 간 어텐션이 켜진 ATTN_ON
조건에서만 의미가 있으므로 self_only=False로 고정한다.
"""
import os, numpy as np, pandas as pd
from _common import load_model, load_test, all_instruments
from attn_patch import install_patch
from paired_eval import paired_eval
from hubs import compute_centrality, top_hubs
from run_experiment import sample_corrupt_set

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_HERE, 'results')


def main(universe='csi300', prefix='opensource'):
    os.makedirs(_RESULTS, exist_ok=True)
    dl_test = load_test(universe, prefix)
    insts = all_instruments(dl_test)

    ks      = [0.05, 0.10, 0.20]
    sigma   = 2.0
    ctype   = 'gaussian'
    seeds   = [0, 1, 2, 3, 4]
    self_only = False                  # 허브는 ATTN_ON에서만 의미

    model = load_model(universe, prefix, seed=0)
    sattn = install_patch(model)

    # 1) 중심성(받은 주목량) 계산 → 허브 랭킹
    print('computing centrality ...')
    centrality = compute_centrality(model, sattn, dl_test)

    p0_cache = {}                      # ATTN_ON clean 예측은 조건 전체에서 공유
    rows = []
    for k in ks:
        hub_set = top_hubs(centrality, k, insts)
        for seed in seeds:
            # 허브: 집합 고정, 노이즈 seed만 변동
            res_h, p0_cache = paired_eval(model, sattn, dl_test, hub_set, ctype,
                                          sigma, seed, self_only, p0_cache)
            rows.append(dict(universe=universe, k=k, mode='hub', seed=seed, **res_h))
            print(rows[-1])

            # 무작위: 집합·노이즈 모두 seed로 변동
            rand_set = sample_corrupt_set(insts, k, seed)
            res_r, p0_cache = paired_eval(model, sattn, dl_test, rand_set, ctype,
                                          sigma, seed, self_only, p0_cache)
            rows.append(dict(universe=universe, k=k, mode='random', seed=seed, **res_r))
            print(rows[-1])

    df = pd.DataFrame(rows)
    out = os.path.join(_RESULTS, f'hubs_{universe}.csv')
    df.to_csv(out, index=False)
    print('saved', out)

    # 요약: k·mode별 dP / clean_IC 평균
    summary = df.groupby(['k', 'mode'])[['clean_IC', 'dP']].mean()
    print('\n=== summary (mean over seeds) ===')
    print(summary)
    return df


if __name__ == '__main__':
    main('csi300', 'opensource')
