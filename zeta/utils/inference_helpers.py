from math import ceil

import torch
import torch.functional as F
from torch import nn


def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float("-inf")
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


def top_k(logits, thres=0.9):
    k = ceil((1 - thres) * logits.shape[-1])
    val, ind, = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs

def top_a(
        logits,
        min_p_pow=2.0,
        min_p_ratio=0.02
):
    probs = F.softmax(logits, dim=-1)
    limit = torch.pow(torch.max(probs), min_p_pow) * min_p_ratio

    logits[probs < limit] = float("-inf")
    logits[probs >= limit] = 1
    return logits

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumnel_sample(t, temperature=1., dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


class ContrastiveTopK(nn.Module):
    def __init__(self, 
                 alpha,
                 k):
        super(ContrastiveTopK, self).__init__()
        self.alpha = alpha
        self.k = k
    
    def top_k(self, logits):
        k = ceil((1 - self.alpha) * logits.shape[-1])
        val, ind = torch.topk(logits, k)
        
        probs = torch.full_like(logits, float('-inf'))
        probs.scatter_(1, ind, val)

        return probs
    
    def forward(self,
                logits_exp,
                logits_ama):
        logits_exp_topk = self.top_k(logits_exp)
        logits_ama_topk = self.top_k(logits_ama)

        #probabilities
        p_exp = F.softmax(logits_exp_topk, dim=-1)
        p_ama = F.softmax(logits_ama_topk, dim=-1)

        #mask
        _, ind = torch.topk(p_exp, self.k)
        mask = torch.zeros_like(p_exp)
        mask.scatter_(1, ind, p_exp[ind] >= self.alpha * p_exp[ind[-1]])

        #scores
        scores = torch.where(mask.bool(), torch.log(p_exp / (p_ama + 1e-8)), 
                             torch.tensor(-float('inf')))
        
        return scores

#alpha = 0.5
#k = 10
# cdk = ContrastiveTopK(alpha, k)

#logits_exp = torch.randn(100, 50)
#logits_ama = torch.randn(100, 50)

#scores
#scores = cdk(logits_exp, logits_ama)
#return