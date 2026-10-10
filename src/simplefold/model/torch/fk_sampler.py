"""Frozen-model Feynman--Kac steering of the existing EM transition.

Particles must share conditioning. The caller supplies an eval-mode model and
three independent RNG streams: initial noise, Brownian increments, resampling.
There are no parameter updates, gradients, reward gradients, or denoiser calls
at off-grid times. Rewards use the model's coordinate units (not necessarily A).

Incremental potentials are exp(beta * (R_k - R_previous)), with an initial
exp(beta * R_0) correction. Their product along any surviving lineage is thus
exp(beta * R_final), even for different initial states. Multinomial SMC is a
finite-particle approximation, not exact independent reward-tilted sampling.
"""

import math

import torch

from .sampler import EMSampler, center_random_augmentation


def make_initial_noise(shape, *, generator, device="cpu", dtype=torch.float32,
                       common_start=False):
    """Draw initial noise with its own RNG; optionally share one starting state."""
    if len(shape) != 3 or shape[0] < 1 or shape[-1] != 3:
        raise ValueError("shape must be (positive number of particles, atoms, 3)")
    drawn_shape = (1, *shape[1:]) if common_start else shape
    noise = torch.randn(drawn_shape, generator=generator, device=device, dtype=dtype)
    return noise.expand(shape).clone() if common_start else noise


class FKSampler(EMSampler):
    """Difference-potential particle selection without modifying the base drift.

    checkpoint_indices are source-state indices in the inherited EM time grid.
    State 0 is always scored for the initial correction; state N is always scored
    as a finished structure. Only requested intermediate checkpoints can
    resample. ess_threshold is a fraction of particle count; 0 disables
    resampling. Decisions require ESS strictly below the threshold (with a small
    numerical tolerance). All conditioning tensors whose leading dimension is
    particle count must be identical across particles; this permits shared
    features and cached model conditioning to remain unchanged after selection.

    score_fn(clean_estimate, time) returns one finite scalar per particle.
    Inputs are detached copies, so an in-place scorer cannot change the sampler.
    Optional checkpoint_fn(index, time, source_state) receives detached copies
    before selection at requested checkpoints only. Its particle indexing is
    the history event's preselection indexing; history parents identify which
    saved states became descendants. Finished states are returned separately.
    """

    def __init__(self, *, beta=1.0, checkpoint_indices=(), ess_threshold=0.8,
                 **kwargs):
        super().__init__(**kwargs)
        self.beta = float(beta)
        self.checkpoint_indices = frozenset(checkpoint_indices)
        self.ess_threshold = float(ess_threshold)

    @staticmethod
    def _reward(score_fn, estimate, t):
        rewards = torch.as_tensor(
            score_fn(estimate.detach().clone(), float(t)),
            device=estimate.device, dtype=torch.float64,
        ).detach()
        if not torch.isfinite(rewards).all():
            raise ValueError("score_fn returned NaN or Inf rewards")
        return rewards

    @torch.no_grad()
    def sample(self, model_fn, flow, noise, batch, *, score_fn,
               brownian_generator, resampling_generator, checkpoint_fn=None):
        if brownian_generator is resampling_generator:
            raise ValueError("Brownian and resampling generators must be separate objects")
        if noise.ndim != 3 or noise.shape[0] < 1 or noise.shape[-1] != 3:
            raise ValueError("noise must have shape (particles, atoms, 3)")
        count = noise.shape[0]
        mask = batch["atom_pad_mask"]
        if mask.shape != noise.shape[:2] or torch.any(mask.sum(dim=1) <= 0):
            raise ValueError("atom_pad_mask must match noise and include occupied atoms")
        steps = self.steps.to(noise.device)
        if torch.any(steps[1:] <= steps[:-1]):
            raise ValueError("time grid must be strictly increasing; check t_start/log_timesteps")
        y = noise.detach().clone()
        log_weights = torch.zeros(count, dtype=torch.float64, device=y.device)
        previous_rewards = torch.zeros_like(log_weights)
        lineage = torch.arange(count, device=y.device)
        history = []
        log_normalizer = 0.0

        def selection_event(rewards, index, t, terminal=False):
            nonlocal log_weights, previous_rewards, lineage, log_normalizer
            log_weights = log_weights + self.beta * (rewards - previous_rewards)
            previous_rewards = rewards
            weights = torch.softmax(log_weights, dim=0)
            ess = float(1.0 / torch.sum(weights.square()))
            allowed = (not terminal and index in self.checkpoint_indices
                       and float(t) < self.w_cutoff and self.tau > 0
                       and float(self.diffusion_coefficient(t)) > 0)
            selected = allowed and ess < self.ess_threshold * count - 1e-10
            parents = (torch.multinomial(weights, count, replacement=True,
                                       generator=resampling_generator)
                       if selected else torch.arange(count, device=y.device))
            history.append({
                "step": index, "time": float(t), "terminal": terminal,
                "rewards": rewards.cpu().tolist(), "log_weights": log_weights.cpu().tolist(),
                "weights": weights.cpu().tolist(), "ess": ess,
                "resampling_allowed": allowed, "resampled": selected,
                "parents": parents.cpu().tolist(), "lineage_before": lineage.cpu().tolist(),
                "path_log_potentials": (self.beta * rewards).cpu().tolist(),
            })
            if selected:
                log_normalizer += float(torch.logsumexp(log_weights, dim=0) - math.log(count))
                previous_rewards = rewards[parents]
                lineage = lineage[parents]
                log_weights = torch.zeros_like(log_weights)
            return parents

        for index in range(self.num_timesteps):
            t, t_next = steps[index], steps[index + 1]
            dt = t_next - t
            # Same centering, model evaluation, score/drift and noise scaling as
            # EMSampler.euler_maruyama_step. Only RNG source and selection differ.
            y = center_random_augmentation(y, mask, augmentation=False, centering=True)
            velocity = model_fn(noised_pos=y, t=t.expand(count), feats=batch)["predict_velocity"]
            if checkpoint_fn is not None and index in self.checkpoint_indices:
                checkpoint_fn(index, float(t), y.detach().clone())
            if index == 0 or index in self.checkpoint_indices:
                rewards = self._reward(score_fn, y + (1.0 - t) * velocity, t)
                parents = selection_event(rewards, index, t)
                y, velocity = y[parents], velocity[parents]
            score = flow.compute_score_from_velocity(velocity, y, t)
            diff_coeff = self.diffusion_coefficient(t)
            drift = velocity + diff_coeff * score
            eps = torch.randn(y.shape, dtype=y.dtype, device=y.device,
                              generator=brownian_generator)
            y = y + drift * dt + torch.sqrt(2.0 * dt * diff_coeff * self.tau) * eps

        # At t=1 the state itself is the finished structure: do not call the
        # denoiser or singular linear-path score there.
        final_rewards = self._reward(score_fn, y, steps[-1])
        selection_event(final_rewards, self.num_timesteps, steps[-1], terminal=True)
        log_normalizer += float(torch.logsumexp(log_weights, dim=0) - math.log(count))
        return {
            "denoised_coords": y.detach(), "final_rewards": final_rewards,
            "log_weights": log_weights, "weights": torch.softmax(log_weights, dim=0),
            "initial_ancestor": lineage, "history": history,
            "path_log_potentials": self.beta * final_rewards,
            "log_normalizer_estimate": log_normalizer,
            "rng_seeds": {"brownian": brownian_generator.initial_seed(),
                          "resampling": resampling_generator.initial_seed()},
        }
