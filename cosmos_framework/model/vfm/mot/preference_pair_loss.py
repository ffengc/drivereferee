# SPDX-License-Identifier: Apache-2.0
# Written by Fengcheng Yu, 2026.

"""Reference-free pairwise preference loss for referee distillation.

``L_pref = -log sigmoid(beta * (L_loser - L_winner))`` over samples that carry a
``loser_action`` tensor. The loser is noised with the winner's sigma schedule and
denoised in a second pass that reuses the winner pass's vision/text tokens.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def extract_loser_action_databatch(
    data_batch: dict,
    to_kwargs: dict,
) -> list[torch.Tensor | None] | None:
    """Extract ``loser_action`` aligned 1:1 with the dense action list.

    Samples whose ``action`` is None are dropped with the same predicate as the dense
    action list. Returns None when the key is absent or no sample carries a loser.
    """
    raw = data_batch.get("loser_action", None)
    if raw is None:
        return None
    actions = data_batch.get("action", None)
    if actions is None:
        return None

    def _unwrap(v):
        while isinstance(v, (list, tuple)):
            if len(v) != 1:
                raise ValueError(f"expected single-item wrapping for loser_action, got len={len(v)}")
            v = v[0]
        return v

    aligned: list[torch.Tensor | None] = []
    for a, l in zip(actions, raw, strict=True):
        if _unwrap(a) is None:
            continue
        loser = _unwrap(l)
        aligned.append(loser.to(**to_kwargs) if loser is not None else None)
    if all(x is None for x in aligned):
        return None
    return aligned


def noise_loser_actions(
    x0_loser: list[torch.Tensor],
    sigmas_action: list[torch.Tensor],
    rectified_flow_action,
    raw_action_dim: list[torch.Tensor] | None,
    tensor_kwargs: dict,
    tensor_kwargs_fp32: dict,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Noise loser trajectories with the winner pass's (condition-masked) sigmas.

    Mirrors the action branch of ``_add_noise_to_input``: fresh epsilon, xt cast to
    the model dtype, padded dims zeroed.
    """
    epsilon = [torch.randn(x0.size(), **tensor_kwargs_fp32) for x0 in x0_loser]
    xt, vt = rectified_flow_action.get_interpolation(epsilon, x0_loser, sigmas_action)
    xt = [xt_i.to(**tensor_kwargs) for xt_i in xt]
    for i in range(len(xt)):
        if raw_action_dim is not None and raw_action_dim[i] is not None:
            xt[i][:, raw_action_dim[i] :] = 0
    return xt, vt


def pair_preference_loss(
    per_instance_winner: torch.Tensor,
    per_instance_loser: torch.Tensor,
    pair_mask: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bracket loss averaged over pair samples only.

    Returns ``(loss, mean_bracket)``; the second value is the detached mean of
    ``L_loser - L_winner`` over pairs, for logging.
    """
    if pair_mask.sum() == 0:
        zero = 0.0 * per_instance_winner.sum()
        return zero, zero.detach()
    bracket = per_instance_loser[pair_mask] - per_instance_winner[pair_mask]
    loss = F.softplus(-beta * bracket).mean()  # -log sigmoid(beta * bracket)
    return loss, bracket.detach().mean()
