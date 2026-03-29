"""
Beta Mixture Model (BMM) for Hard Negative Mining.

Implements ProGCL-style BMM-based hard negative weighting for document-level
relation extraction. The key insight is that in a batch of negatives, some are
"false negatives" (similar to the anchor in embedding space) and should be
down-weighted; true negatives have low similarity and should be kept.

References:
    - ProGCL: Prototypical Graph Contrastive Learning (Wang et al., 2022)
    - Beta Mixture Models for noisy label detection
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Beta

logger = logging.getLogger(__name__)

# Numerical stability constants
_EPS = 1e-6
_LOG_EPS = 1e-8


class BetaMixtureModel:
    """
    Two-component Beta Mixture Model fitted via the EM algorithm.

    Used to separate similarity scores into two clusters:
      - Component 0: "true negatives"  — low similarity (alpha < beta)
      - Component 1: "false negatives" — high similarity (alpha > beta)

    Parameters
    ----------
    num_components : int
        Number of mixture components (fixed at 2 for the current design).
    max_em_iters : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance on log-likelihood change.
    """

    def __init__(
        self,
        num_components: int = 2,
        max_em_iters: int = 10,
        tol: float = 1e-4,
    ) -> None:
        self.num_components = num_components
        self.max_em_iters = max_em_iters
        self.tol = tol

        # Component parameters: shape [num_components]
        self.alphas: Tensor = torch.tensor([2.0, 5.0])  # [K]
        self.betas: Tensor = torch.tensor([5.0, 2.0])   # [K]
        self.pis: Tensor = torch.tensor([0.5, 0.5])     # [K] mixing weights

        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _beta_log_pdf(self, x: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
        """
        Compute log-PDF of Beta(alpha, beta) evaluated at x.

        Parameters
        ----------
        x : Tensor
            Values in (0, 1), shape [N].
        alpha : Tensor
            Concentration parameter, scalar.
        beta : Tensor
            Concentration parameter, scalar.

        Returns
        -------
        Tensor
            Log-PDF values, shape [N].
        """
        # Clamp x away from boundaries to avoid log(0)
        x_clamped = x.clamp(_LOG_EPS, 1.0 - _LOG_EPS)
        dist = Beta(alpha, beta)
        return dist.log_prob(x_clamped)  # [N]

    def _compute_responsibilities(self, s: Tensor) -> Tensor:
        """
        E-step: compute posterior responsibilities gamma[k, n].

        gamma[k, n] = pi_k * p(s_n | theta_k) / sum_k pi_k * p(s_n | theta_k)

        Parameters
        ----------
        s : Tensor
            Similarity scores, shape [N].

        Returns
        -------
        Tensor
            Responsibilities, shape [K, N].
        """
        device = s.device
        K = self.num_components

        alphas = self.alphas.to(device)   # [K]
        betas  = self.betas.to(device)    # [K]
        pis    = self.pis.to(device)      # [K]

        # log_probs[k, n] = log pi_k + log p(s_n | k)
        log_probs = torch.zeros(K, s.shape[0], device=device)  # [K, N]
        for k in range(K):
            log_probs[k] = torch.log(pis[k] + _EPS) + self._beta_log_pdf(
                s, alphas[k], betas[k]
            )

        # Numerically stable softmax over K
        log_probs_norm = log_probs - torch.logsumexp(log_probs, dim=0, keepdim=True)
        gamma = torch.exp(log_probs_norm)  # [K, N]
        return gamma

    def _moments_to_beta_params(
        self, mean: Tensor, var: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Convert mean and variance to Beta distribution alpha/beta via method of moments.

        alpha = mean * (mean*(1-mean)/var - 1)
        beta  = (1-mean) * (mean*(1-mean)/var - 1)

        Parameters
        ----------
        mean : Tensor
            Weighted mean, scalar.
        var : Tensor
            Weighted variance, scalar.

        Returns
        -------
        Tuple[Tensor, Tensor]
            (alpha, beta) scalars, clamped to [0.1, 100].
        """
        var_clamped = var.clamp(min=_EPS)
        concentration = mean * (1.0 - mean) / var_clamped - 1.0
        concentration = concentration.clamp(min=_EPS)  # must be positive
        alpha = mean * concentration
        beta  = (1.0 - mean) * concentration
        alpha = alpha.clamp(0.1, 100.0)
        beta  = beta.clamp(0.1, 100.0)
        return alpha, beta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, similarities: Tensor) -> None:
        """
        Fit the 2-component Beta Mixture Model on similarity scores via EM.

        Parameters
        ----------
        similarities : Tensor
            Cosine similarity scores in [0, 1], shape [N].
            Values are clamped into (eps, 1-eps) for Beta distribution stability.
        """
        s = similarities.detach().float()

        # --- Edge cases ---
        if s.numel() < 4:
            # Too few samples to fit reliably — keep defaults
            logger.debug("BMM.fit: too few samples (%d), keeping defaults.", s.numel())
            return

        if s.max() - s.min() < _EPS:
            # All similarities are the same — degenerate
            logger.debug("BMM.fit: all similarities identical (%.4f), keeping defaults.", s.mean().item())
            return

        # Clamp into valid Beta support
        s = s.clamp(_LOG_EPS, 1.0 - _LOG_EPS)
        N = s.shape[0]
        device = s.device

        # Initialize parameters on the correct device
        self.alphas = torch.tensor([2.0, 5.0], device=device)
        self.betas  = torch.tensor([5.0, 2.0], device=device)
        self.pis    = torch.tensor([0.5, 0.5], device=device)

        prev_log_lik = -float("inf")

        for iteration in range(self.max_em_iters):
            # ---- E-step ----
            gamma = self._compute_responsibilities(s)  # [K, N]

            # ---- M-step ----
            n_k = gamma.sum(dim=1)  # [K] effective counts

            new_alphas = torch.zeros(self.num_components, device=device)
            new_betas  = torch.zeros(self.num_components, device=device)
            new_pis    = n_k / (N + _EPS)  # [K]

            for k in range(self.num_components):
                w = gamma[k]                          # [N] responsibilities for component k
                w_sum = n_k[k].clamp(min=_EPS)

                mean_k = (w * s).sum() / w_sum        # scalar
                var_k  = (w * (s - mean_k) ** 2).sum() / w_sum  # scalar
                var_k  = var_k.clamp(min=_EPS)

                alpha_k, beta_k = self._moments_to_beta_params(mean_k, var_k)
                new_alphas[k] = alpha_k
                new_betas[k]  = beta_k

            self.alphas = new_alphas
            self.betas  = new_betas
            self.pis    = new_pis.clamp(_EPS, 1.0)  # ensure valid mixing weights

            # ---- Convergence check via log-likelihood ----
            log_probs = torch.zeros(self.num_components, N, device=device)
            for k in range(self.num_components):
                log_probs[k] = torch.log(self.pis[k] + _EPS) + self._beta_log_pdf(
                    s, self.alphas[k], self.betas[k]
                )
            log_lik = torch.logsumexp(log_probs, dim=0).sum().item()

            if abs(log_lik - prev_log_lik) < self.tol:
                logger.debug("BMM converged at iteration %d.", iteration + 1)
                break
            prev_log_lik = log_lik

        self._is_fitted = True
        logger.debug(
            "BMM fitted: alpha=[%.3f, %.3f] beta=[%.3f, %.3f] pi=[%.3f, %.3f]",
            self.alphas[0].item(), self.alphas[1].item(),
            self.betas[0].item(),  self.betas[1].item(),
            self.pis[0].item(),    self.pis[1].item(),
        )

    def predict_true_negative_prob(self, similarities: Tensor) -> Tensor:
        """
        Compute the posterior probability that each sample belongs to the
        "true negative" component (the one with the lower mean).

        Parameters
        ----------
        similarities : Tensor
            Similarity scores in [0, 1], shape [N].

        Returns
        -------
        Tensor
            p(true_negative | s) for each sample, shape [N].
        """
        s = similarities.detach().float()
        device = s.device

        if not self._is_fitted:
            # Before fitting, treat everything as equally likely true negative
            return torch.ones(s.shape[0], device=device) * 0.5

        s = s.clamp(_LOG_EPS, 1.0 - _LOG_EPS)

        # Identify the "true negative" component by lower mean
        # mean of Beta(alpha, beta) = alpha / (alpha + beta)
        means = self.alphas.to(device) / (
            self.alphas.to(device) + self.betas.to(device) + _EPS
        )  # [K]
        true_neg_component = int(means.argmin().item())

        # Compute full responsibilities
        gamma = self._compute_responsibilities(s)  # [K, N]
        p_true_neg = gamma[true_neg_component]      # [N]
        return p_true_neg

    @property
    def component_means(self) -> Tensor:
        """Return the mean of each Beta component. Shape: [K]."""
        return self.alphas / (self.alphas + self.betas + _EPS)


class HardNegativeWeighter:
    """
    Wrapper around BetaMixtureModel that manages warm-up, periodic re-fitting,
    and produces per-negative weights for contrastive training.

    During warm-up (epoch < bmm_warmup_epochs), uniform weights are returned so
    that the model is not distracted by an untrained BMM.

    After warm-up, the BMM is re-fitted every ``update_every_n_steps`` steps on
    the current batch's similarities, and posterior probabilities are returned as
    weights.

    Parameters
    ----------
    bmm_warmup_epochs : int
        Number of initial epochs where uniform weights are used.
    update_every_n_steps : int
        Re-fit the BMM every N gradient steps.
    temperature : float
        Cosine similarity temperature (informational; not used directly here).
    """

    def __init__(
        self,
        bmm_warmup_epochs: int = 3,
        update_every_n_steps: int = 100,
        temperature: float = 0.07,
    ) -> None:
        self.bmm_warmup_epochs = bmm_warmup_epochs
        self.update_every_n_steps = update_every_n_steps
        self.temperature = temperature

        self.bmm = BetaMixtureModel()
        self.current_epoch: int = 0
        self.step_count: int = 0
        self.is_warmed_up: bool = False

    def compute_weights(
        self,
        anchor_embs: Tensor,
        negative_embs: Tensor,
        epoch: int,
        step: int,
    ) -> Tensor:
        """
        Compute per-negative importance weights for contrastive loss.

        Higher weight ≡ more likely a *true* negative ≡ safe to push away harder.
        False negatives (high similarity but same class) are down-weighted.

        Parameters
        ----------
        anchor_embs : Tensor
            Anchor embeddings, shape [A, dim].
        negative_embs : Tensor
            Negative embeddings, shape [M, dim].  All negatives are shared
            across all anchors (in-batch negatives style).
        epoch : int
            Current training epoch (0-indexed).
        step : int
            Current global training step.

        Returns
        -------
        Tensor
            Weights, shape [A, M].  Values in (0, 1].
        """
        self.current_epoch = epoch
        self.step_count = step

        A = anchor_embs.shape[0]
        M = negative_embs.shape[0]
        device = anchor_embs.device

        # ---- Warm-up: return uniform weights ----
        if epoch < self.bmm_warmup_epochs:
            return torch.ones(A, M, device=device)

        self.is_warmed_up = True

        # ---- Compute cosine similarities [A, M] ----
        # Normalize for cosine similarity
        a_norm = F.normalize(anchor_embs.detach().float(), p=2, dim=-1)   # [A, dim]
        n_norm = F.normalize(negative_embs.detach().float(), p=2, dim=-1) # [M, dim]
        cosine_sim = torch.mm(a_norm, n_norm.t())  # [A, M]

        # Map cosine similarity from [-1, 1] → [0, 1] for Beta support
        sim_01 = (cosine_sim + 1.0) / 2.0  # [A, M]

        # ---- Periodically re-fit the BMM ----
        if step % self.update_every_n_steps == 0:
            flat_sims = sim_01.reshape(-1)  # [A*M]
            self.bmm.fit(flat_sims)

        # ---- Compute p(true_negative | s) ----
        flat_sims = sim_01.reshape(-1)                                    # [A*M]
        flat_weights = self.bmm.predict_true_negative_prob(flat_sims)     # [A*M]
        weights = flat_weights.reshape(A, M)                              # [A, M]

        # Ensure weights are on the correct device and dtype
        weights = weights.to(device=device, dtype=anchor_embs.dtype)

        return weights
