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
