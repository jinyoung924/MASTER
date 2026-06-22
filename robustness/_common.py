# robustness/_common.py
import os, sys, pickle, torch

# --- repo-root bootstrap ---------------------------------------------------
# `python robustness/sanity.py` puts robustness/ on sys.path but NOT the repo
# root, so `from master import ...` would fail. Add the parent dir explicitly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# ---------------------------------------------------------------------------

import pandas as pd, numpy as np
from master import MASTERModel

CFG = dict(d_feat=158, d_model=256, t_nhead=4, s_nhead=2,
           T_dropout_rate=0.5, S_dropout_rate=0.5,
           gate_input_start_index=158, gate_input_end_index=221,
           n_epochs=1, lr=1e-5, GPU=0, train_stop_loss_thred=0.95)


def load_model(universe='csi300', prefix='opensource', seed=0,
               model_dir=None, beta=None):
    if model_dir is None:
        model_dir = os.path.join(_REPO_ROOT, 'model')
    if beta is None:
        beta = 5 if universe == 'csi300' else 2
    m = MASTERModel(beta=beta, seed=seed, save_path=model_dir,
                    save_prefix=universe, **CFG)
    ckpt = os.path.join(model_dir, f'{universe}_{prefix}_{seed}.pkl')
    m.load_param(ckpt)            # sets self.fitted = 'Previously trained.' (a string!)
    m.fitted = 0                  # ◀ 필수 우회: load_param 직후 predict()의 `if self.fitted<0`가
                                  #    '문자열 < 정수' 비교라 TypeError로 죽는다(레포 버그). 0으로 덮어쓴다.
    m.model.eval()
    return m


def load_test(universe='csi300', prefix='opensource', data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(_REPO_ROOT, 'data')
    p = os.path.join(data_dir, prefix, f'{universe}_dl_test.pkl')
    with open(p, 'rb') as f:
        return pickle.load(f)


def daily_instruments(dl_test):
    """sampler와 동일 순서로 (datetime, [instruments]) 리스트 반환."""
    idx = dl_test.get_index()                    # MultiIndex; 실제 레벨 순서는 (instrument, datetime)
    df = idx.to_frame(index=False)               # 레벨 이름을 컬럼명으로 보존(위치 기반 rename 금지)
    assert {'datetime', 'instrument'} <= set(df.columns), \
        f'unexpected index levels: {list(df.columns)}'
    # DailyBatchSamplerRandom은 groupby(datetime)로 일자 카운트(기본 정렬) → 동일 재현.
    # groupby는 그룹 내 원순서를 보존하므로 일자별 instrument 순서가 sampler와 일치한다.
    groups = []
    for dt, sub in df.groupby('datetime', sort=True):
        groups.append((dt, sub['instrument'].tolist()))
    return groups   # i번째 원소가 i번째 배치에 대응


def all_instruments(dl_test):
    return sorted({s for _, insts in daily_instruments(dl_test) for s in insts})
