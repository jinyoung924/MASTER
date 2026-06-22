import numpy as np, torch
from _common import load_model, load_test, daily_instruments, all_instruments
from attn_patch import install_patch
from paired_eval import paired_eval


def gate_reproduce(universe='csi300', prefix='opensource', tol=0.01):
    """게이트1: 무오염+ON IC가 performance.xlsx 해당 seed0과 근접.
    목표값(opensource 데이터 기준, performance.xlsx에서 확인):
      - csi300 + opensource ckpt:  IC≈0.0653, RankIC≈0.0689  (seed0)
      - csi300 + original  ckpt:   IC≈0.0678, RankIC≈0.0722  (seed0)
      - csi800 + opensource ckpt:  IC≈0.0458, RankIC≈0.0569  (seed0)
    ※ 논문 Table1(IC 0.064 / RankIC 0.076)과 비교하지 말 것. 오픈소스 데이터는
       논문 원본(회사 비공개) 데이터와 값이 달라 RankIC가 특히 차이난다.
    """
    m = load_model(universe, prefix, 0)
    dl = load_test(universe, prefix)
    _, metrics = m.predict(dl)
    print('reproduced:', metrics)   # 위 목표값과 |IC차| < tol 인지 확인
    return metrics


def gate_self_only_zero(universe='csi300', prefix='opensource'):
    """게이트2: self_only에서 깨끗한 종목 ΔP≈0."""
    m = load_model(universe, prefix, 0); s = install_patch(m)
    dl = load_test(universe, prefix)
    insts = sorted({x for _, l in daily_instruments(dl) for x in l})
    cset = set(insts[:max(1, len(insts)//5)])    # 20% 오염
    res, _ = paired_eval(m, s, dl, cset, 'gaussian', 2.0, 0, self_only=True)
    print('self_only dP =', res['dP'])
    assert res['dP'] < 1e-5, 'SELF_ONLY leaks → masking/alignment bug'


def gate_channel_invariance():
    """게이트4: 오염이 [158:]·라벨 미접촉."""
    from corruptors import corrupt_gaussian
    x = torch.randn(10, 8, 222)
    feat = x[:, :, :-1].clone()
    out = corrupt_gaussian(feat, torch.tensor([0, 1, 2]), sigma=2.0,
                           gen=torch.Generator().manual_seed(0))
    assert torch.allclose(out[:, :, 158:], feat[:, :, 158:]), 'market channels changed!'
    print('channel invariance OK')


if __name__ == '__main__':
    gate_channel_invariance()
    gate_reproduce()
    gate_self_only_zero()
