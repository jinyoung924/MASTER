import os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_HERE, 'results')


def rel_drop(df, value='clean_IC'):
    base = (df[df.k == 0].groupby('self_only')[value].mean())
    df = df.copy()
    df['rel_drop'] = df.apply(lambda r: 1 - r[value]/base[r.self_only], axis=1)
    return df


def main(csv=None):
    if csv is None:
        csv = os.path.join(_RESULTS, 'main_csi300.csv')
    df = pd.read_csv(csv)
    # 'ctype' 컬럼이 없던 구버전 CSV도 처리
    ctypes = sorted(df['ctype'].unique()) if 'ctype' in df.columns else [None]

    for ctype in ctypes:
        sub = df if ctype is None else df[df.ctype == ctype]
        suffix = '' if ctype is None else f'_{ctype}'
        title = 'Inter-stock contamination (CSI300)' if ctype is None \
            else f'Inter-stock contamination — {ctype} (CSI300)'
        g = sub.groupby(['self_only', 'k'])

        # 그림1: rel_drop(clean_IC)
        d = rel_drop(sub).groupby(['self_only', 'k'])['rel_drop'].agg(['mean', 'std']).reset_index()
        plt.figure()
        for so, lab in [(False, 'ATTN_ON'), (True, 'SELF_ONLY')]:
            s = d[d.self_only == so]
            plt.errorbar(s.k*100, s['mean'], yerr=s['std'], marker='o', label=lab)
        plt.xlabel('corrupted stocks k (%)'); plt.ylabel('relative clean-IC drop')
        plt.legend(); plt.title(title)
        plt.savefig(os.path.join(_RESULTS, f'fig1_rel_drop{suffix}.png'), dpi=150, bbox_inches='tight')

        # 그림2: dP
        d2 = g['dP'].agg(['mean', 'std']).reset_index()
        plt.figure()
        for so, lab in [(False, 'ATTN_ON'), (True, 'SELF_ONLY')]:
            s = d2[d2.self_only == so]
            plt.errorbar(s.k*100, s['mean'], yerr=s['std'], marker='s', label=lab)
        plt.xlabel('corrupted stocks k (%)'); plt.ylabel('clean-stock |Δprediction|')
        plt.legend(); plt.title('Contamination signal magnitude' + ('' if ctype is None else f' — {ctype}'))
        plt.savefig(os.path.join(_RESULTS, f'fig2_dP{suffix}.png'), dpi=150, bbox_inches='tight')

    # 그림2b: gaussian vs missing의 ATTN_ON dP 직접 비교
    if 'ctype' in df.columns and len(ctypes) > 1:
        plt.figure()
        on = df[df.self_only == False]
        for ctype in ctypes:
            s = on[on.ctype == ctype].groupby('k')['dP'].agg(['mean', 'std']).reset_index()
            plt.errorbar(s.k*100, s['mean'], yerr=s['std'], marker='s', label=ctype)
        plt.xlabel('corrupted stocks k (%)'); plt.ylabel('clean-stock |Δprediction|')
        plt.legend(); plt.title('Contamination type comparison (ATTN_ON, CSI300)')
        plt.savefig(os.path.join(_RESULTS, 'fig2c_type_compare.png'), dpi=150, bbox_inches='tight')


def plot_hubs(csv=None):
    """C3: 허브 vs 무작위 오염의 dP / clean_IC 비교."""
    if csv is None:
        csv = os.path.join(_RESULTS, 'hubs_csi300.csv')
    df = pd.read_csv(csv)

    for value, fname, ylab in [
        ('dP', 'fig3_hubs_dP.png', 'clean-stock |Δprediction|'),
        ('clean_IC', 'fig4_hubs_IC.png', 'clean-stock IC')]:
        d = df.groupby(['mode', 'k'])[value].agg(['mean', 'std']).reset_index()
        plt.figure()
        for mode, lab in [('hub', 'HUB'), ('random', 'RANDOM')]:
            s = d[d['mode'] == mode]
            plt.errorbar(s.k*100, s['mean'], yerr=s['std'], marker='o', label=lab)
        plt.xlabel('corrupted stocks k (%)'); plt.ylabel(ylab)
        plt.legend(); plt.title('Hub vs random contamination (CSI300)')
        plt.savefig(os.path.join(_RESULTS, fname), dpi=150, bbox_inches='tight')


def plot_combined(csv=None):
    """보고서용 결합 그림.
    fig1: ATTN_ON에서 relative clean-IC drop을 gaussian vs missing 한 그림에.
    fig2: dP를 SELF_ONLY(대조) + ATTN_ON gaussian + ATTN_ON missing 세 계열로 한 그림에.
    """
    if csv is None:
        csv = os.path.join(_RESULTS, 'main_csi300.csv')
    df = pd.read_csv(csv)
    on = df[df.self_only == False].copy()
    ctypes = sorted(on['ctype'].unique())
    base = on[on.k == 0]['clean_IC'].mean()        # ATTN_ON 무오염 기준선(=0.0651)
    on['rel_drop'] = 1 - on['clean_IC'] / base

    # --- fig1: rel_drop, SELF_ONLY(대조) + ATTN_ON gaussian/missing ---
    # 세 계열 모두 같은 ATTN_ON 무오염 baseline으로 정규화(공통 분모) →
    # SELF_ONLY의 base≈0으로 인한 오차막대 폭발 없이 평평한 대조선이 됨.
    off = df[df.self_only == True].copy()
    off['rel_drop'] = 1 - off['clean_IC'] / base
    plt.figure()
    so = off.groupby('k')['rel_drop'].agg(['mean', 'std']).reset_index()
    plt.errorbar(so.k*100, so['mean'], yerr=so['std'], marker='^', linestyle='--',
                 color='gray', label='SELF_ONLY (control)')
    d = on.groupby(['ctype', 'k'])['rel_drop'].agg(['mean', 'std']).reset_index()
    for ctype in ctypes:
        s = d[d.ctype == ctype]
        plt.errorbar(s.k*100, s['mean'], yerr=s['std'], marker='o', label=f'ATTN_ON {ctype}')
    plt.xlabel('corrupted stocks k (%)'); plt.ylabel('relative clean-IC drop (vs ATTN_ON base)')
    plt.legend(); plt.title('Relative clean-IC drop by corruption type (CSI300)')
    plt.savefig(os.path.join(_RESULTS, 'fig1_rel_drop_combined.png'), dpi=150, bbox_inches='tight')

    # --- fig2: dP, SELF_ONLY + ATTN_ON gaussian + ATTN_ON missing ---
    plt.figure()
    so = df[df.self_only == True].groupby('k')['dP'].agg(['mean', 'std']).reset_index()
    plt.errorbar(so.k*100, so['mean'], yerr=so['std'], marker='^', linestyle='--',
                 color='gray', label='SELF_ONLY (control)')
    for ctype in ctypes:
        s = on[on.ctype == ctype].groupby('k')['dP'].agg(['mean', 'std']).reset_index()
        plt.errorbar(s.k*100, s['mean'], yerr=s['std'], marker='s', label=f'ATTN_ON {ctype}')
    plt.xlabel('corrupted stocks k (%)'); plt.ylabel('clean-stock |Δprediction|')
    plt.legend(); plt.title('Contamination signal magnitude (CSI300)')
    plt.savefig(os.path.join(_RESULTS, 'fig2_dP_combined.png'), dpi=150, bbox_inches='tight')


if __name__ == '__main__':
    main()
    plot_combined()
