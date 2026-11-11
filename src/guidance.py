"""Autograd reward guidance for PepTune's discrete masked diffusion.

Requires the SAME vocabulary/order/tokenization for diffusion and reward
models. If tokens differ, train an additional differentiable student instead.
"""
from __future__ import annotations

from collections.abc import Mapping
import torch
import torch.nn.functional as F
from torch import nn

from soft_xgb import SoftXGBoost


class DifferentiablePeptideRewards(nn.Module):
    """Frozen PeptideCLM encoder followed by the four soft XGBoost predictors.

    Input is [B,L,V] *probabilities* aligned to the reward encoder's SMILES
    vocabulary. They must cover CLS and SEP exactly as original tokenization.
    """
    def __init__(self, peptideclm_roformer: nn.Module,
                 predictors: Mapping[str, SoftXGBoost]):
        super().__init__()
        self.encoder = peptideclm_roformer.eval()
        self.encoder.requires_grad_(False)  # still differentiable through inputs
        self.predictors = nn.ModuleDict(predictors)
        for p in self.predictors.values():
            p.eval().requires_grad_(False)

    def forward(self, token_probs: torch.Tensor, attention_mask: torch.Tensor,
                *, mode="soft") -> dict[str, torch.Tensor]:
        if token_probs.ndim != 3:
            raise ValueError("token_probs must have dimensions [batch, length, vocab]")
        w = self.encoder.get_input_embeddings().weight
        if token_probs.shape[-1] != w.shape[0]:
            raise ValueError("Diffusion/reward vocabulary sizes do not match")
        # Input embeddings in the encoder's own vocabulary. No NumPy/no detach.
        expected = token_probs.to(w.dtype) @ w
        hidden = self.encoder(inputs_embeds=expected,
                              attention_mask=attention_mask).last_hidden_state
        # Exact PepTune behavior: unmasked mean, INCLUDING special/padded tokens.
        features = hidden.mean(dim=1).to(torch.float32)
        features = torch.nan_to_num(features)
        result = {key: predictor(features, mode=mode)
                  for key, predictor in self.predictors.items()}
        if "hemolysis" in result:
            result["hemolysis"] = 1 - result["hemolysis"]
        return result


def make_token_probs(log_probs: torch.Tensor, current_ids: torch.Tensor,
                     mask_token_id: int, *, temp: float = 1.0) -> torch.Tensor:
    """Soft candidate completion for masked sites, exact one-hot elsewhere."""
    if temp <= 0:
        raise ValueError("temp must be positive")
    vocab = log_probs.shape[-1]
    # Model probabilities may already forbid MASK; maintain that constraint.
    proposal = torch.softmax(log_probs / temp, dim=-1)
    fixed = F.one_hot(current_ids, num_classes=vocab).to(dtype=proposal.dtype)
    editable = (current_ids == mask_token_id).unsqueeze(-1)
    return torch.where(editable, proposal, fixed)


def guided_log_probs(
    log_probs: torch.Tensor, current_ids: torch.Tensor,
    attention_mask: torch.Tensor, rewards: DifferentiablePeptideRewards,
    *, mask_token_id: int, weights: Mapping[str, float], strength: float = 1.0,
    reward_temperature: float = 1.0, gradient_clip: float = 5.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """First-order reward tilt: q(k) ~ p(k) * exp(strength * dR/dp_k).

    Takes the log-probabilities returned by Diffusion.forward(...), with
    SUBS masking already applied. Returns new log-probabilities. Not a sampler.
    Use inside torch.enable_grad() even when the sampling loop is no_grad.
    """
    if strength < 0 or gradient_clip <= 0:
        raise ValueError("strength must be >=0 and gradient_clip >0")
    with torch.enable_grad():
        base = log_probs.detach().to(dtype=torch.float32)
        token_probs = make_token_probs(base, current_ids, mask_token_id,
                                       temp=reward_temperature).detach().requires_grad_(True)
        scores = rewards(token_probs, attention_mask)
        names = [name for name, weight in weights.items() if weight != 0]
        if not names:
            return base, {name: value.detach() for name, value in scores.items()}
        reward = sum(float(weights[name]) * scores[name] for name in names)
        gradient = torch.autograd.grad(reward.sum(), token_probs)[0]
        gradient = gradient.clamp(-gradient_clip, gradient_clip)
        # Never allow modifications to already unmasked tokens.
        editable = (current_ids == mask_token_id) & attention_mask.bool()
        adjusted = base + strength * gradient * editable.unsqueeze(-1)
        # Maintain all forbidden tokens, including MASK, from original SUBS.
        forbidden = base <= -1e5
        adjusted = adjusted.masked_fill(forbidden, -1e6)
        adjusted = torch.log_softmax(adjusted, dim=-1)
        return adjusted.detach(), {name: value.detach() for name, value in scores.items()}
