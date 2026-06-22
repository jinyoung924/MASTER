import os, sys

# --- repo-root bootstrap ---------------------------------------------------
# Ensure the repo root is importable before `from base_model import ...`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# ---------------------------------------------------------------------------

import numpy as np, pandas as pd, torch
from base_model import calc_ic
from _common import daily_instruments
from corruptors import CORRUPTORS


@torch.no_grad()
def paired_eval(model, sattn, dl_test, corrupt_set, corrupt_type,
                sigma, seed, self_only, p0_cache=None):
    """
    corrupt_set: set(instrument)  # 오염 종목 집합(전 기간 고정)
    반환: dict(clean_IC, clean_RankIC, dP), 그리고 p0_cache
    """
    sattn.self_only = self_only
    sattn.capture = False
    model.model.eval()
    device = model.device
    gen = torch.Generator().manual_seed(seed)

    loader = model._init_data_loader(dl_test, shuffle=False, drop_last=False)
    days = daily_instruments(dl_test)
    corruptor = CORRUPTORS[corrupt_type]

    ics, rics, dps = [], [], []
    if p0_cache is None:
        p0_cache = {}

    for b, data in enumerate(loader):
        data = torch.squeeze(data, dim=0)
        dt, insts = days[b]
        assert data.shape[0] == len(insts), \
            f'alignment broken at batch {b}: N={data.shape[0]} vs {len(insts)}'

        feature = data[:, :, 0:-1].to(device)            # (N, T, 221)
        label = data[:, -1, -1].numpy()

        clean_rows = np.array([i for i, s in enumerate(insts) if s not in corrupt_set])
        corr_rows  = np.array([i for i, s in enumerate(insts) if s in corrupt_set])
        if len(clean_rows) < 5:        # 평가 표본 너무 적은 날 skip
            continue

        # Run-0 (clean) — 모델조건별로 캐시 (오염 무관, 결정적)
        key = (dt, self_only)
        if key in p0_cache:
            p0 = p0_cache[key]
        else:
            p0 = model.model(feature.float()).cpu().numpy()
            p0_cache[key] = p0

        # Run-1 (corrupted)
        if len(corr_rows) > 0 and corrupt_type is not None:
            feat_c = corruptor(feature.cpu(), torch.as_tensor(corr_rows),
                               sigma=sigma, gen=gen).to(device)
            p1 = model.model(feat_c.float()).cpu().numpy()
        else:
            p1 = p0

        # 깨끗한 종목만 평가
        ic, ric = calc_ic(p1[clean_rows], label[clean_rows])
        ics.append(ic); rics.append(ric)
        dps.append(np.mean(np.abs(p1[clean_rows] - p0[clean_rows])))

    return (dict(clean_IC=np.nanmean(ics),
                 clean_RankIC=np.nanmean(rics),
                 dP=np.nanmean(dps)),
            p0_cache)
