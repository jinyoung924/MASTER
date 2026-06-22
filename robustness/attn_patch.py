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
