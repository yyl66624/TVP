"""NLPrompt-inspired training-free aggregation.

OT_CAM replaces the per-token softmax in TF_CAM with an entropy-regularised
optimal-transport assignment between spatial tokens and cluster prototypes.
Compared with softmax:
  * the cluster marginal is constrained (balanced cluster usage, no collapse);
  * an optional dustbin column lets noisy / transient patches (cars, people)
    leave the descriptor (PromptOT-style purification);
  * cost clipping bounds the influence of a single over-confident token
    (PromptMAE-style robust influence).

Everything is computed at inference time: no training, no labels.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.free_module import TF_CAM


class SoftmaxTokenCAM(TF_CAM):
    """TF_CAM with attention normalised over tokens (dim=2) instead of clusters.

    The upstream code normalises over clusters (`dim=1`), so the pooled value of
    each cluster is not a convex combination of tokens.  Normalising over tokens
    turns each cluster into a proper attention-pooling head.
    """

    def __init__(self, clusters_num=16, dim=768, normalize_input=True,
                 work_with_tokens=False, temperature=1.0):
        super().__init__(clusters_num=clusters_num, dim=dim,
                         normalize_input=normalize_input,
                         work_with_tokens=work_with_tokens)
        self.temperature = temperature

    def forward(self, x):
        if self.work_with_tokens:
            x = x.permute(0, 2, 1)
            N, D, _ = x.shape[:]
        else:
            N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)
        x_flat = x.reshape(N, D, -1)
        logit_input = F.normalize(x, p=2, dim=1) if getattr(self, "norm_first", False) else x
        attention = self.conv(logit_input).reshape(N, self.clusters_num, -1)
        attention = F.softmax(attention / self.temperature, dim=2)
        attn_out = torch.einsum("ncl,ndl->ncd", attention, x_flat)

        attn_out = F.normalize(attn_out, p=2, dim=2)
        attn_out = attn_out.reshape(N, -1)
        attn_out = F.normalize(attn_out, p=2, dim=1)
        return attn_out


class OT_CAM(TF_CAM):
    """Drop-in replacement for TF_CAM using Sinkhorn assignments."""

    def __init__(self, clusters_num=16, dim=768, normalize_input=True,
                 work_with_tokens=False, ot_reg=0.2, ot_iters=30, ot_temp=1.0,
                 dustbin=0.0, dustbin_cost=0.0, cost_clip=None):
        super().__init__(clusters_num=clusters_num, dim=dim,
                         normalize_input=normalize_input,
                         work_with_tokens=work_with_tokens)
        self.ot_reg = ot_reg
        self.ot_iters = ot_iters
        self.ot_temp = ot_temp
        self.dustbin = dustbin
        self.dustbin_cost = dustbin_cost
        self.cost_clip = cost_clip

    @staticmethod
    def _sinkhorn(logits, reg, iters, dustbin=0.0, dustbin_cost=0.0):
        """logits: (B, L, C) assignment scores.

        Returns transport plan (B, L, C) whose columns sum to (1-dustbin)/C
        (or a (B, L, C+1) plan if dustbin > 0, the last column being the bin).
        """
        B, L, C = logits.shape
        if dustbin > 0:
            bin_col = torch.full((B, L, 1), dustbin_cost,
                                 dtype=logits.dtype, device=logits.device)
            logits = torch.cat([logits, bin_col], dim=2)
            C2 = C + 1
            b = torch.full((C2,), (1.0 - dustbin) / C, dtype=logits.dtype,
                           device=logits.device)
            b[-1] = dustbin
        else:
            C2 = C
            b = torch.full((C,), 1.0 / C, dtype=logits.dtype, device=logits.device)
        a = torch.full((L,), 1.0 / L, dtype=logits.dtype, device=logits.device)

        logP = -logits / reg
        log_a = torch.log(a)[None, :, None]
        log_b = torch.log(b)[None, None, :]
        for _ in range(iters):
            logP = logP - torch.logsumexp(logP, dim=2, keepdim=True) + log_a
            logP = logP - torch.logsumexp(logP, dim=1, keepdim=True) + log_b
        return torch.exp(logP)

    def forward(self, x):
        if self.work_with_tokens:
            x = x.permute(0, 2, 1)
            N, D, _ = x.shape[:]
            x_spatial = x
        else:
            N, D, H, W = x.shape[:]
            x_spatial = x

        # cluster assignment scores from the k-means prototypes
        logit_input = F.normalize(x_spatial, p=2, dim=1) if getattr(self, "norm_first", False) else x_spatial
        logits = self.conv(logit_input).view(N, self.clusters_num, -1)
        logits = logits.permute(0, 2, 1)  # (N, L, C)
        if self.cost_clip is not None:
            logits = self.cost_clip * torch.tanh(logits / self.cost_clip)
        logits = logits / self.ot_temp

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)
        x_flat = x.reshape(N, D, -1)

        plan = self._sinkhorn(logits, self.ot_reg, self.ot_iters,
                              self.dustbin, self.dustbin_cost)  # (N,L,C(+1))
        if self.dustbin > 0:
            plan = plan[:, :, :-1]
        # column-normalise so each cluster pools with weights summing to one
        plan = plan / (plan.sum(dim=1, keepdim=True) + 1e-8)
        attn_out = torch.einsum("nlc,ndl->ncd", plan, x_flat)

        attn_out = F.normalize(attn_out, p=2, dim=2)
        attn_out = attn_out.reshape(N, -1)
        attn_out = F.normalize(attn_out, p=2, dim=1)
        return attn_out
