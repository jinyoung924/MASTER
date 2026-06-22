# 방향 1 세부 구현서 — 데이터·추론·비교 (3부작 중 3번)

> **문서 성격**: 실제 코드 수준의 구현 명세. 동반 문서 ②의 설계를 그대로 코드로 옮긴다.
> 아래 코드는 레포(`SJTU-DMTai/MASTER`)의 실제 인터페이스(`master.py`, `base_model.py`)에 맞춰 작성한 **참조 구현**이다. 데이터로 실제 실행해 §9 새너티 체크를 통과시킨 뒤 본 실험에 사용한다.

---

## 0. 디렉터리 구조 (산출물 배치)

```
MASTER/                     # 원본 레포 (clone)
├── master.py  base_model.py  main.py
├── model/   csi300_opensource_0.pkl ...   # 체크포인트(레포 포함)
├── data/    opensource/csi300_dl_test.pkl ...   # 별도 다운로드 필요(§1)
└── robustness/             # ◀ 우리가 추가하는 실험 코드
    ├── corruptors.py
    ├── attn_patch.py
    ├── paired_eval.py
    ├── hubs.py
    ├── run_experiment.py
    ├── plots.py
    ├── sanity.py
    └── results/            # CSV·그림 출력
```

---

## 1. 환경 · 의존성 · 데이터

### 1.1 의존성
```bash
pip install "pandas==1.5.3" "torch==1.11.0" pyqlib matplotlib scipy
```
- `pyqlib`는 **필수**: 공개 `*.pkl`은 qlib 데이터셋 객체로 직렬화돼 있어, 언피클 시 qlib이 임포트 가능해야 한다.
- GPU 없으면 자동 CPU 폴백(코드상 `cuda if available else cpu`). CSI300 추론은 CPU로도 가능.

### 1.2 데이터 다운로드 (중요)
- 레포에는 **체크포인트와 `csi_market_information.csv`만** 있고, 학습/검증/테스트 `*.pkl`은 없다. README의 OneDrive/MEGA/Baidu 링크에서 받아 `data/opensource/`(또는 `data/original/`)에 푼다.
- **체크포인트-데이터 소스 짝을 반드시 맞춘다**: `csi300_opensource_0.pkl` ↔ opensource 데이터, `csi300_original_0.pkl` ↔ original 데이터. 섞으면 무오염 IC가 어긋난다.
- **Qlib 기본 데이터로 대체하지 말 것**: Qlib 디폴트 데이터는 기간·시장지수 구성(CSI100/300/500)이 달라 게이팅 입력이 본 레포(CSI300/500/800)와 불일치한다. 반드시 레포가 게시한 데이터를 쓴다.
- **예상 사항**: 공개 데이터는 하루치에 전 종목의 약 95%만 포함한다(원본은 100%). 이는 정상이며, 페어드 비교에서 상쇄된다. 무오염 IC 재현 목표도 이 95% 데이터로 측정된 `performance.xlsx` 수치다(§9 게이트1).

### 1.3 경로/OS 패치
- `main.py`는 Windows 역슬래시(`data\opensource\...`)를 쓴다. Linux/Mac에서는 `os.path.join` 또는 슬래시로 바꿔 로드한다. 우리 실험 코드는 처음부터 `os.path.join`을 쓴다.

### 1.4 모델 로드 헬퍼 (모든 스크립트 공용)
```python
# robustness/_common.py
import os, pickle, torch
from master import MASTERModel

CFG = dict(d_feat=158, d_model=256, t_nhead=4, s_nhead=2,
           T_dropout_rate=0.5, S_dropout_rate=0.5,
           gate_input_start_index=158, gate_input_end_index=221,
           n_epochs=1, lr=1e-5, GPU=0, train_stop_loss_thred=0.95)

def load_model(universe='csi300', prefix='opensource', seed=0,
               model_dir='model', beta=None):
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

def load_test(universe='csi300', prefix='opensource', data_dir='data'):
    p = os.path.join(data_dir, prefix, f'{universe}_dl_test.pkl')
    with open(p, 'rb') as f:
        return pickle.load(f)
```

---

## 2. 데이터 파이프라인 사실 (코드 인덱싱의 근거)

- DataLoader는 `DailyBatchSamplerRandom(shuffle=False)`로 **하루 = 1 배치**를 datetime 정렬 순으로 산출.
- 한 배치 `data`는 `squeeze(dim=0)` 후 `(N, T, F=222)`.
- **채널 맵 (절대 기준)**:
  - `data[:, :, 0:158]` → 종목 factor (**오염 대상**)
  - `data[:, :, 158:221]` → 시장정보 63개 (전 종목 동일, **불가침**)
  - `data[:, -1, 221]` → 라벨 (**불가침**)
- 모델 입력은 `feature = data[:, :, 0:221]`(라벨 제외), 예측은 `model.model(feature.float())` → `(N,)` 벡터.
- 날짜별 instrument 정렬은 `dl_test.get_index()`(MultiIndex)에서 복원한다. sampler와 동일한 정렬(datetime group, 그룹 내 원순서)을 재현해야 종목 인덱싱이 어긋나지 않는다.

### 2.1 날짜별 instrument 리스트 복원
```python
# robustness/_common.py (계속)
import pandas as pd, numpy as np

def daily_instruments(dl_test):
    """sampler와 동일 순서로 (datetime, [instruments]) 리스트 반환."""
    idx = dl_test.get_index()                    # MultiIndex(datetime, instrument)
    df = idx.to_frame(index=False)
    df.columns = ['datetime', 'instrument']
    # DailyBatchSamplerRandom은 groupby(datetime)로 일자 카운트(기본 정렬) → 동일 재현
    groups = []
    for dt, sub in df.groupby('datetime', sort=True):
        groups.append((dt, sub['instrument'].tolist()))
    return groups   # i번째 원소가 i번째 배치에 대응
```
> **정합성 가정**: 공개 데이터는 "chronically gathered then grouped by date"이므로 위 정렬이 sampler와 일치한다. §9의 종목-정렬 게이트로 매 배치 `N == len(instruments)`를 확인한다.

---

## 3. 모듈 1 — 오염 연산자 `corruptors.py`

```python
import torch

FACTOR_END = 158   # [0:158]만 오염

def _slice(feat):                 # feat: (N, T, 221)
    return feat[..., :FACTOR_END]

def corrupt_gaussian(feat, rows, sigma, gen):
    out = feat.clone()
    noise = torch.randn(out[rows, :, :FACTOR_END].shape, generator=gen) * sigma
    out[rows, :, :FACTOR_END] = (out[rows, :, :FACTOR_END] + noise).clamp(-3, 3)
    return out

def corrupt_missing(feat, rows, **_):
    out = feat.clone()
    out[rows, :, :FACTOR_END] = 0.0
    return out

def corrupt_shock(feat, rows, gen, **_):
    out = feat.clone()
    sign = torch.randint(0, 2, out[rows, :, :FACTOR_END].shape, generator=gen) * 2 - 1
    out[rows, :, :FACTOR_END] = 3.0 * sign.float()
    return out

CORRUPTORS = {'gaussian': corrupt_gaussian,
              'missing':  corrupt_missing,
              'shock':    corrupt_shock}
```
- `rows`는 **그 배치 안에서 오염 대상 종목의 로컬 인덱스**(§5에서 instrument→로컬 매핑).
- `gen`은 `torch.Generator`로 seed 고정.
- 모든 연산자가 `[158:]`을 건드리지 않음(§9 채널 불변 게이트로 검증).

---

## 4. 모듈 2 — `SAttention` 패치 `attn_patch.py`

`SAttention.forward`를 self-only/캡처 기능이 있는 버전으로 **런타임 치환**(레포 파일 미수정).

```python
import math, torch
from torch.nn import functional as F

def find_sattention(model):
    cands = [m for m in model.model.modules() if type(m).__name__ == 'SAttention']
    assert len(cands) == 1, f'expected 1 SAttention, got {len(cands)}'
    return cands[0]

def patched_forward(self, x):
    # x: (N, T, D)
    x = self.norm1(x)
    q = self.qtrans(x).transpose(0, 1)   # (T, N, D)
    k = self.ktrans(x).transpose(0, 1)
    v = self.vtrans(x).transpose(0, 1)
    dim = int(self.d_model / self.nhead)

    att_output, captured = [], []
    for i in range(self.nhead):
        sl = slice(i*dim, None) if i == self.nhead-1 else slice(i*dim, (i+1)*dim)
        qh, kh, vh = q[:, :, sl], k[:, :, sl], v[:, :, sl]
        T_, N_, _ = qh.shape
        if getattr(self, 'self_only', False):
            A = torch.eye(N_, device=qh.device).unsqueeze(0).expand(T_, N_, N_)
        else:
            A = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
        # eval()에서는 dropout이 identity라 결과 불변(학습엔 사용 금지)
        if self.attn_dropout:
            A = self.attn_dropout[i](A)
        att_output.append(torch.matmul(A, vh).transpose(0, 1))   # (N, T, dim)
        if getattr(self, 'capture', False):
            captured.append(A.detach())                          # (T, N, N)
    if getattr(self, 'capture', False):
        # 헤드 평균 attention 저장 (행 i=query, 열 j=key)
        self._last_attn = torch.stack(captured, 0).mean(0)       # (T, N, N)

    att_output = torch.concat(att_output, dim=-1)
    xt = x + att_output
    xt = self.norm2(xt)
    return xt + self.ffn(xt)

def install_patch(model):
    sattn = find_sattention(model)
    import types
    sattn.forward = types.MethodType(patched_forward, sattn)
    sattn.self_only = False
    sattn.capture = False
    sattn._last_attn = None
    return sattn

# 사용:
#   sattn = install_patch(model)
#   sattn.self_only = True/False   # 모델 조건 토글
#   sattn.capture   = True/False   # 중심성 계산 시
```
> 정확성: 단위행렬 `A`에서는 `A @ vh = vh` → 각 종목이 자기 value만 받음 = 종목 간 정보 차단. softmax 분기와 코드 경로가 동일해, self_only=False면 원본과 수치적으로 일치해야 한다(§9 게이트 1로 검증).

---

## 5. 모듈 3 — 페어드 추론·지표 `paired_eval.py`

설계 ②의 §2(반사실)·§5(지표)를 구현. 핵심은 **깨끗한 종목만** 평가하고, **Run-0를 캐시**하는 것.

```python
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
```
주의:
- `calc_ic`는 pandas corr이라 라벨 NaN은 자동 무시.
- `p0_cache`는 `(datetime, self_only)`로 키. 한 데이터셋·모델조건에서 모든 k·유형·seed에 재사용 → 연산 대폭 절감.
- k=0(빈 `corrupt_set`)이면 `p1=p0` → `dP=0`, `clean_IC=` 무오염 기준선.

---

## 6. 모듈 4 — 허브 식별 `hubs.py` (C3, 선택)

```python
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
```
- 허브 오염 vs 무작위 오염 비교: 같은 k에서 `corrupt_set`을 `top_hubs(...)` 대 무작위 추출로 바꿔 `paired_eval` 실행 후 `clean_IC` 비교.

---

## 7. 모듈 5 — 실험 러너 `run_experiment.py`

```python
import os, itertools, random, numpy as np, pandas as pd
from _common import load_model, load_test, daily_instruments
from attn_patch import install_patch
from paired_eval import paired_eval

def all_instruments(dl_test):
    return sorted({s for _, insts in daily_instruments(dl_test) for s in insts})

def sample_corrupt_set(universe_list, frac, seed):
    rng = random.Random(seed)
    k = int(round(frac * len(universe_list)))
    return set(rng.sample(universe_list, k))

def main(universe='csi300', prefix='opensource'):
    os.makedirs('robustness/results', exist_ok=True)
    dl_test = load_test(universe, prefix)
    insts = all_instruments(dl_test)

    ks      = [0.0, 0.05, 0.10, 0.20, 0.40]
    types   = ['gaussian']            # 메인. 이후 ['gaussian','missing','shock']
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
    df.to_csv(f'robustness/results/main_{universe}.csv', index=False)

if __name__ == '__main__':
    main('csi300', 'opensource')
```
- **가중치 seed vs 오염 seed 구분**: 모델 가중치는 항상 `*_0.pkl`(seed0 체크포인트). `seed`는 *오염 종목 추출·노이즈*의 무작위성만 제어한다. (체크포인트가 seed별로 더 있으면 모델 seed도 추가 축으로 확장 가능.)

---

## 8. 모듈 6 — 플로팅 `plots.py`

```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt

def rel_drop(df, value='clean_IC'):
    base = (df[df.k == 0].groupby('self_only')[value].mean())
    df = df.copy()
    df['rel_drop'] = df.apply(lambda r: 1 - r[value]/base[r.self_only], axis=1)
    return df

def main(csv='robustness/results/main_csi300.csv'):
    df = pd.read_csv(csv)
    g = df.groupby(['self_only', 'k'])

    # 그림1: rel_drop(clean_IC)
    d = rel_drop(df).groupby(['self_only','k'])['rel_drop'].agg(['mean','std']).reset_index()
    plt.figure()
    for so, lab in [(False,'ATTN_ON'), (True,'SELF_ONLY')]:
        s = d[d.self_only==so]
        plt.errorbar(s.k*100, s['mean'], yerr=s['std'], marker='o', label=lab)
    plt.xlabel('corrupted stocks k (%)'); plt.ylabel('relative clean-IC drop')
    plt.legend(); plt.title('Inter-stock contamination (CSI300)')
    plt.savefig('robustness/results/fig1_rel_drop.png', dpi=150, bbox_inches='tight')

    # 그림2: dP
    d2 = g['dP'].agg(['mean','std']).reset_index()
    plt.figure()
    for so, lab in [(False,'ATTN_ON'), (True,'SELF_ONLY')]:
        s = d2[d2.self_only==so]
        plt.errorbar(s.k*100, s['mean'], yerr=s['std'], marker='s', label=lab)
    plt.xlabel('corrupted stocks k (%)'); plt.ylabel('clean-stock |Δprediction|')
    plt.legend(); plt.title('Contamination signal magnitude')
    plt.savefig('robustness/results/fig2_dP.png', dpi=150, bbox_inches='tight')

if __name__ == '__main__':
    main()
```

---

## 9. 새너티 체크 `sanity.py` (실험 전 반드시 통과)

설계 ②의 §10 게이트를 자동화.

```python
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
    out = corrupt_gaussian(feat, torch.tensor([0,1,2]), sigma=2.0,
                           gen=torch.Generator().manual_seed(0))
    assert torch.allclose(out[:, :, 158:], feat[:, :, 158:]), 'market channels changed!'
    print('channel invariance OK')

if __name__ == '__main__':
    gate_channel_invariance()
    gate_reproduce()
    gate_self_only_zero()
```
**게이트 3(종목 정렬)**은 `paired_eval` 내부 `assert data.shape[0]==len(insts)`로 상시 강제됨.

---

## 10. 실행 순서 요약

1. 데이터 다운로드 → `data/opensource/` 배치(§1.2).
2. `python robustness/sanity.py` → **게이트 1·2·4 통과 확인**(여기서 막히면 멈추고 디버그).
3. `python robustness/run_experiment.py` → `results/main_csi300.csv` 생성.
4. `python robustness/plots.py` → 대표 그림 2장.
5. (선택) `types`를 3종으로 확장, `hubs.py`로 C3, `universe='csi800'`로 재현.

---

## 11. 예상 연산량

- CSI300 test ≈ 수백 일 × N≈300. forward 1회는 매우 가벼움.
- 메인 스윕(2 조건 × 5 k × 5 seed, p0 캐시 적용) ≈ 수십 회 평가 패스. GPU 수 분, CPU 수십 분 규모.
- p0 캐시가 ON/OFF당 1회만 clean forward를 돌리므로 corrupted forward만 반복 → 절반 이하로 절감.

---

## 12. 흔한 함정 체크리스트

- [ ] Windows 역슬래시 경로 → `os.path.join`로 교체했는가.
- [ ] 체크포인트와 데이터 소스(original/opensource) 짝을 맞췄는가.
- [ ] 오염을 `[0:158]`에만 적용했는가(게이트4).
- [ ] 평가를 **깨끗한 종목**으로 한정했는가(오염 종목 포함 금지).
- [ ] `rel_drop`으로 ON/OFF를 각자 k=0 기준 정규화해 비교했는가.
- [ ] seed가 *오염 무작위성*만 제어하고 모델 가중치는 고정인가.
- [ ] self_only ΔP≈0(게이트2)를 본 실험 전에 확인했는가.
