import numpy as np, torch
from collections import defaultdict
from _common import daily_instruments


@torch.no_grad()
def compute_centrality(model, sattn, dl_test):
    sattn.self_only = False
    sattn.capture = True
    model.model.eval()
    loader = model._init_data_loader(dl_test, shuffle=False, drop_last=False)
    days = daily_instruments(dl_test)

    recv_sum = defaultdict(float); recv_cnt = defaultdict(int)
    for b, data in enumerate(loader):
        data = torch.squeeze(data, dim=0)
        _, insts = days[b]
        feat = data[:, :, 0:-1].to(model.device)
        _ = model.model(feat.float())            # forward → sattn._last_attn 채움
        A = sattn._last_attn                      # (T, N, N), 행=query 열=key
        recv = A.mean(0).sum(0).cpu().numpy()     # 종목별 받은 주목량(열 합, T평균)
        for s, r in zip(insts, recv):
            recv_sum[s] += float(r); recv_cnt[s] += 1
    centrality = {s: recv_sum[s]/recv_cnt[s] for s in recv_sum}
    return centrality   # 높을수록 허브


def top_hubs(centrality, frac, all_instruments):
    k = int(round(frac * len(all_instruments)))
    ranked = sorted(centrality, key=centrality.get, reverse=True)
    return set(ranked[:k])
