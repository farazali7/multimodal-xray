import torch
import math

from src.utils.constants import EPS


def top_k(logits: torch.Tensor, thresh: float = 0.9):
    k = math.ceil((1 - thresh) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)

    return probs


def log(t):
    return torch.log(t.clamp(min=EPS))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)
