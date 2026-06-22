# MASTER 강건성 검증 — Inter-Stock 오염 전이(Contamination Transfer)

주가 예측 모델 **MASTER**(*Market-Guided Stock Transformer*, AAAI-24)의 핵심 기여인 **종목 간 어텐션(inter-stock attention)** 이 강점인 동시에 강건성의 취약 통로가 될 수 있는지를 **추론 전용(inference-only)** 으로 검증하는 실험 코드입니다.

> 본 저장소는 원본 [SJTU-DMTai/MASTER](https://github.com/SJTU-DMTai/MASTER)를 기반으로 하며, 추가된 실험 코드는 모두 [`robustness/`](robustness/) 폴더에 있습니다. 모델 자체는 **재학습하지 않고** 사전학습 체크포인트를 그대로 사용합니다.

---

## 1. 실험 한눈에 보기

한 거래일의 입력에서 **일부 종목(k%)의 고유 특징만 오염**시킨 뒤, *입력이 전혀 바뀌지 않은 깨끗한 종목들*의 예측이 어떻게 변하는지를 측정합니다.

- **페어드 반사실 비교:** 같은 날짜를 두 번 추론합니다 — Run-0(원본)과 Run-1(오염). 깨끗한 종목의 입력은 양쪽이 동일하므로, 예측 차이(`ΔP`)는 *오염된 다른 종목을 거쳐 전파된 양*만을 의미합니다.
- **self-only 통제군:** 종목 간 어텐션을 단위행렬로 치환(각 종목이 자기 자신만 참조 = per-stock 모델 등가)하면 전파 경로가 끊깁니다. 여기서 `ΔP≈0`이면 전파 경로가 어텐션임이 입증됩니다.
- **검증 클레임:**
  - **C1 (현상):** 일부 오염이 깨끗한 종목의 IC/RankIC를 떨어뜨리는가?
  - **C2 (경로):** 그 원인이 종목 간 어텐션인가? (self-only 통제로 식별)
  - **C3 (중심성):** 어텐션 상 주목을 많이 받는 *허브 종목*을 오염시키면 피해가 더 큰가?

자세한 배경·설계·구현·결과는 다음 문서를 참고하세요.

| 문서 | 내용 |
|---|---|
| [`01_claim_report.md`](01_claim_report.md) | 핵심 클레임과 검증 논리 |
| [`02_experiment_design.md`](02_experiment_design.md) | 실험 설계(변수·지표·통제) |
| [`03_implementation_spec.md`](03_implementation_spec.md) | 코드 수준 구현 명세 |
| [`04_results_report.md`](04_results_report.md) | 실험 결과 분석 |
| [`05_최종결과보고서.md`](05_최종결과보고서.md) | 논문 형식 최종 보고서 |

---

## 2. 디렉터리 구조

```
MASTER/
├── master.py  base_model.py  main.py     # 원본 MASTER 모델 코드
├── model/                                 # 사전학습 체크포인트 (저장소에 포함)
│   └── csi300_opensource_0.pkl ...
├── data/
│   ├── csi_market_information.csv         # 저장소에 포함
│   └── opensource/                        # ◀ 직접 다운로드 필요 (gitignore 처리됨, §4)
│       └── csi300_dl_test.pkl ...
└── robustness/                            # ◀ 본 실험 코드
    ├── corruptors.py     # 오염 연산자 (gaussian / missing)
    ├── attn_patch.py     # SAttention self-only / 캡처 패치
    ├── paired_eval.py    # 페어드 추론 + clean-IC / ΔP 계산
    ├── hubs.py           # 어텐션 중심성(허브) 식별
    ├── run_experiment.py # 메인 스윕 러너
    ├── run_hubs.py       # 허브 vs 무작위 비교 (C3)
    ├── plots.py          # 그림 생성
    ├── sanity.py         # 새너티 게이트 (실험 전 필수)
    └── results/          # CSV·그림 출력 (저장소에 포함)
```

---

## 3. 환경 설정

```bash
git clone https://github.com/jinyoung924/MASTER.git
cd MASTER

# 원본 명세와 동일한 버전 (권장)
pip install "pandas==1.5.3" "torch==1.11.0" pyqlib matplotlib scipy
```

- `pyqlib`는 **필수**입니다. 공개 데이터(`*.pkl`)가 qlib 데이터셋 객체로 직렬화되어 있어, 언피클 시 qlib이 임포트 가능해야 합니다.
- GPU가 없어도 됩니다(코드가 자동으로 CPU로 폴백). CSI300 추론은 CPU로도 수행 가능합니다.

<details>
<summary><b>Apple Silicon(arm64) Mac 사용자 주의</b></summary>

`torch==1.11.0`은 Apple Silicon용 휠이 없습니다(macOS x86_64 휠만 존재). conda로 **x86_64(Rosetta) 환경**을 만들어 설치하세요.

```bash
CONDA_SUBDIR=osx-64 conda create -y -n master_env python=3.8
conda activate master_env
conda config --env --set subdir osx-64
pip install "pandas==1.5.3" "torch==1.11.0" pyqlib matplotlib scipy
```

> 버전 핀을 고집하지 않는다면 최신 torch/pandas(arm64 네이티브)로도 동일한 수치 결과가 재현됩니다(게이트1로 확인됨). 이 경우 더 빠릅니다.
</details>

---

## 4. 데이터 다운로드 (중요)

용량이 큰 opensource 데이터셋(`data/opensource/*.pkl`, 합계 수 GB)은 **저장소에 포함되지 않습니다**(`.gitignore` 처리). 원본 저장소에서 직접 받아야 합니다.

1. 원본 [SJTU-DMTai/MASTER](https://github.com/SJTU-DMTai/MASTER) README의 **데이터 다운로드 링크(OneDrive / MEGA / Baidu)** 로 이동합니다.
2. `*_dl_train.pkl`, `*_dl_valid.pkl`, `*_dl_test.pkl` 파일을 받아 **`data/opensource/`** 폴더에 그대로 넣습니다.
3. 본 실험(CSI300 메인)에는 최소한 `data/opensource/csi300_dl_test.pkl`만 있으면 됩니다.

> ⚠️ **체크포인트–데이터 소스 짝을 맞추세요.** `csi300_opensource_0.pkl` ↔ opensource 데이터 (`model/`의 체크포인트는 저장소에 포함되어 있습니다). Qlib 기본 데이터로 대체하지 마세요 — 기간·시장지수 구성이 달라 게이팅 입력이 불일치합니다.

---

## 5. 실행 방법

모든 명령은 **저장소 루트(`MASTER/`)** 에서 실행합니다.

```bash
# 1) 새너티 게이트 — 실험 전 반드시 통과 확인
#    (무오염 IC 재현 ≈ 0.0651, self-only ΔP=0, 채널 불변)
python robustness/sanity.py

# 2) 메인 스윕 — gaussian+missing × ATTN_ON/SELF_ONLY × k × seed
#    → results/main_csi300.csv
python robustness/run_experiment.py

# 3) 허브 vs 무작위 오염 비교 (C3)
#    → results/hubs_csi300.csv
python robustness/run_hubs.py

# 4) 그림 생성 → results/*.png
python robustness/plots.py
```

> CPU에서 메인 스윕은 수십 분~1시간대가 소요될 수 있습니다(데이터로더 오버헤드가 지배적). 게이트가 막히면 거기서 멈추고 데이터 경로·체크포인트 짝을 먼저 점검하세요.

---

## 6. 주요 결과 요약

| 클레임 | 결과 | 근거 |
|---|---|---|
| **C2** (경로 = 종목 간 어텐션) | **확인 (가장 강함)** | self-only에서 `ΔP=0`, 원본에선 `ΔP>0`이 k에 선형 증가 |
| **C1** (깨끗한 종목 IC 저하) | 현상은 존재하나 강한 형태는 미성립 | IC 저하 ≤4%로 통계적 비유의 → 무작위 오염엔 견고 |
| **C3** (허브 의존성) | **확인** | 허브 오염이 ΔP·IC 모두 더 큰 피해(k=20%에서 유의) |

대표 그림: `robustness/results/fig2_dP_combined.png`(C2), `fig1_rel_drop_combined.png`(C1), `fig3_hubs_dP.png`·`fig4_hubs_IC.png`(C3). 상세 해석은 [`05_최종결과보고서.md`](05_최종결과보고서.md) 참고.

---

## 7. 인용 / 크레딧

본 실험은 원본 MASTER 모델과 공개 데이터·체크포인트에 의존합니다. 모델·데이터를 사용할 경우 원저자의 논문과 저장소를 인용하세요.

```bibtex
@inproceedings{li2024master,
  title={MASTER: Market-Guided Stock Transformer for Stock Price Forecasting},
  author={Li, Tong and Liu, Zhaoyang and Shen, Yanyan and Wang, Xue and Chen, Haokun and Huang, Sen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```

- 원본 코드·데이터: https://github.com/SJTU-DMTai/MASTER
