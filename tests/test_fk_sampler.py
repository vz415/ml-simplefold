"""Tiny CPU tensor checks; never load weights or sample a protein model."""

import sys
import unittest
from pathlib import Path

import torch

# The upstream model uses absolute imports rooted at src/simplefold.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "simplefold"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from model.flow import LinearPath
from model.torch.fk_sampler import FKSampler, make_initial_noise
from model.torch.sampler import EMSampler
from sample_complex_fk import expand_shared_batch


def generator(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


class ToyVelocity:
    def __init__(self):
        self.times = []

    def __call__(self, *, noised_pos, t, feats):
        if torch.is_grad_enabled():
            raise AssertionError("Sampling must not enable gradients")
        self.times.append(float(t[0]))
        return {"predict_velocity": .15 * noised_pos.sin() + t[:, None, None] * .07}


class FKSamplerTests(unittest.TestCase):
    def setUp(self):
        self.noise = make_initial_noise((5, 4, 3), generator=generator(91), dtype=torch.float64)
        self.batch = {"atom_pad_mask": torch.ones(5, 4, dtype=torch.float64)}
        self.flow = LinearPath()

    def run_sampler(self, sampler, *, model=None, score=None, noise=None, brownian=12,
                    resampling=34, checkpoint=None):
        return sampler.sample(
            model or ToyVelocity(), self.flow, self.noise if noise is None else noise,
            self.batch, score_fn=score or (lambda x, t: x[:, 0, 0]),
            brownian_generator=generator(brownian), resampling_generator=generator(resampling),
            checkpoint_fn=checkpoint,
        )

    def test_zero_beta_is_exact_base_em_for_linear_and_log_grid(self):
        for log_grid in (False, True):
            kwargs = dict(num_timesteps=7, t_start=.03, tau=.01, log_timesteps=log_grid)
            base, steered = EMSampler(**kwargs), FKSampler(beta=0, checkpoint_indices=(0, 2, 5),
                                                         ess_threshold=1, **kwargs)
            # Base EM uses the global RNG; the new sampler uses an isolated RNG
            # with the same draws, so its unchanged transition must match exactly.
            torch.manual_seed(12)
            expected = self.noise.clone()
            model = ToyVelocity()
            for index in range(base.num_timesteps):
                expected = base.euler_maruyama_step(model, self.flow, expected,
                                                   base.steps[index], base.steps[index + 1],
                                                   self.batch)
            result = self.run_sampler(steered)
            torch.testing.assert_close(result["denoised_coords"], expected, rtol=0, atol=0)
            self.assertFalse(any(h["resampled"] for h in result["history"]))
            torch.testing.assert_close(result["weights"], torch.full((5,), .2, dtype=torch.float64))

    def test_difference_potentials_include_initial_correction_and_actual_endpoint(self):
        sampler = FKSampler(num_timesteps=6, t_start=.1, beta=2.3,
                            checkpoint_indices=(1, 3, 5), ess_threshold=0)
        calls = []

        def reward(x, t):
            self.assertFalse(x.requires_grad)
            calls.append((x.clone(), t))
            return x[:, 0].square().sum(dim=1) + 3 * t

        model = ToyVelocity()
        result = self.run_sampler(sampler, model=model, score=reward)
        self.assertEqual(len(model.times), sampler.num_timesteps)
        self.assertLess(max(model.times), 1)
        self.assertEqual([h["step"] for h in result["history"]], [0, 1, 3, 5, 6])
        self.assertEqual(calls[-1][1], 1.)
        torch.testing.assert_close(calls[-1][0], result["denoised_coords"])
        # A missing initial correction would instead leave beta*(R_final-R_0).
        torch.testing.assert_close(result["log_weights"], sampler.beta * result["final_rewards"])
        torch.testing.assert_close(result["weights"], torch.softmax(
            sampler.beta * result["final_rewards"], dim=0))

    def test_resampling_selects_cached_state_and_velocity_then_branches_independently(self):
        noise = torch.zeros_like(self.noise)
        noise[:, 0, 0] = torch.arange(5, dtype=noise.dtype)
        noise[:, 1, 0] = -torch.arange(5, dtype=noise.dtype)
        kwargs = dict(num_timesteps=4, t_start=.1, tau=.02)
        result = self.run_sampler(FKSampler(beta=1000, checkpoint_indices=(0,),
                                           ess_threshold=1, **kwargs),
                                  noise=noise, score=lambda x, t: x[:, 0, 0])
        first = result["history"][0]
        self.assertTrue(first["resampled"])
        self.assertEqual(first["parents"], [4] * 5)
        self.assertEqual(result["initial_ancestor"].tolist(), [4] * 5)
        # Same selected starting coordinates AND their matching velocity are
        # equivalent to beginning an unsteered run with those repeated states.
        expected = self.run_sampler(FKSampler(beta=0, **kwargs),
                                    noise=noise[4:5].expand_as(noise).clone())
        torch.testing.assert_close(result["denoised_coords"], expected["denoised_coords"],
                                   rtol=0, atol=0)
        self.assertGreater(float(torch.std(result["denoised_coords"][:, 0, 0])), 0)
        alternate = self.run_sampler(FKSampler(beta=0, **kwargs),
                                     noise=noise[4:5].expand_as(noise).clone(), brownian=13)
        self.assertFalse(torch.equal(expected["denoised_coords"], alternate["denoised_coords"]))

    def test_weights_and_reward_lineages_remain_consistent_across_resampling(self):
        sampler = FKSampler(num_timesteps=8, t_start=.1, tau=.03, beta=4,
                            checkpoint_indices=(0, 2, 4, 6), ess_threshold=1)
        result = self.run_sampler(sampler, score=lambda x, t: x[:, 0, 0].square() + t)
        old_reward = torch.zeros(5, dtype=torch.float64)
        old_weight = torch.zeros_like(old_reward)
        old_lineage = torch.arange(5)
        for event in result["history"]:
            current_reward = torch.tensor(event["rewards"], dtype=torch.float64)
            current_weight = torch.tensor(event["log_weights"], dtype=torch.float64)
            torch.testing.assert_close(current_weight, old_weight + sampler.beta *
                                       (current_reward - old_reward))
            self.assertEqual(event["lineage_before"], old_lineage.tolist())
            parents = torch.tensor(event["parents"])
            if event["resampled"]:
                old_reward = current_reward[parents]
                old_weight = torch.zeros_like(old_weight)
                old_lineage = old_lineage[parents]
            else:
                old_reward, old_weight = current_reward, current_weight
        torch.testing.assert_close(result["log_weights"], old_weight)
        torch.testing.assert_close(result["initial_ancestor"], old_lineage)

    def test_no_selection_after_stochastic_cutoff_or_with_zero_tau(self):
        for tau in (.01, 0):
            sampler = FKSampler(num_timesteps=4, t_start=.1, tau=tau, beta=100,
                                w_cutoff=.5, checkpoint_indices=(0, 1, 2, 3),
                                ess_threshold=1)
            result = self.run_sampler(sampler)
            for event in result["history"]:
                if event["time"] >= .5 or tau == 0 or event["terminal"]:
                    self.assertFalse(event["resampled"])
                    self.assertFalse(event["resampling_allowed"])
            self.assertEqual(result["history"][-1]["parents"], list(range(5)))

    def test_reproducible_separate_rng_streams_and_common_initial_noise(self):
        common = make_initial_noise(self.noise.shape, generator=generator(91),
                                    common_start=True, dtype=torch.float64)
        torch.testing.assert_close(common, common[:1].expand_as(common))
        sampler = FKSampler(num_timesteps=4, t_start=.1, checkpoint_indices=(1, 2))
        first = self.run_sampler(sampler, noise=common)
        second = self.run_sampler(sampler, noise=common)
        self.assertEqual(first["history"], second["history"])
        torch.testing.assert_close(first["denoised_coords"], second["denoised_coords"], rtol=0, atol=0)
        same = generator(2)
        with self.assertRaisesRegex(ValueError, "separate"):
            sampler.sample(ToyVelocity(), self.flow, common, self.batch,
                           score_fn=lambda x, t: x[:, 0, 0],
                           brownian_generator=same, resampling_generator=same)

    def test_scoring_cannot_mutate_particle_state(self):
        sampler = FKSampler(num_timesteps=3, t_start=.1, beta=0, checkpoint_indices=(1,))
        expected = self.run_sampler(sampler, score=lambda x, t: torch.zeros(5))

        def mutating_score(x, t):
            x.fill_(float("nan"))
            return torch.zeros(5)

        observed = self.run_sampler(sampler, score=mutating_score)
        torch.testing.assert_close(observed["denoised_coords"], expected["denoised_coords"],
                                   rtol=0, atol=0)

    def test_checkpoint_records_preselection_source_state_without_mutating_it(self):
        sampler = FKSampler(num_timesteps=5, t_start=.1, beta=10,
                            checkpoint_indices=(1, 3), ess_threshold=1)
        source_states, callback_states = [], []

        class RecordingModel(ToyVelocity):
            def __call__(self, *, noised_pos, t, feats):
                source_states.append(noised_pos.clone())
                return super().__call__(noised_pos=noised_pos, t=t, feats=feats)

        def callback(index, t, state):
            self.assertFalse(state.requires_grad)
            callback_states.append((index, t, state.clone()))
            state.fill_(float("nan"))

        observed = self.run_sampler(sampler, model=RecordingModel(), checkpoint=callback)
        expected = self.run_sampler(sampler)
        torch.testing.assert_close(observed["denoised_coords"], expected["denoised_coords"],
                                   rtol=0, atol=0)
        self.assertEqual([index for index, _, _ in callback_states], [1, 3])
        for index, t, state in callback_states:
            self.assertEqual(t, float(sampler.steps[index]))
            torch.testing.assert_close(state, source_states[index], rtol=0, atol=0)
            event = next(h for h in observed["history"] if h["step"] == index)
            self.assertEqual(event["time"], t)

    def test_nan_and_inf_rewards_are_rejected(self):
        sampler = FKSampler(num_timesteps=3, t_start=.1)
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                    self.run_sampler(sampler, score=lambda x, t: torch.full((5,), value))

    def test_chunked_velocity_preserves_global_resampling_and_rng(self):
        noise = torch.zeros_like(self.noise)
        noise[:, 0, 0] = torch.arange(5, dtype=noise.dtype)
        noise[:, 1, 0] = -torch.arange(5, dtype=noise.dtype)
        sizes = []

        class ChunkModel(ToyVelocity):
            def __call__(self, *, noised_pos, t, feats):
                sizes.append(noised_pos.shape[0])
                self.assert_shapes(noised_pos, t, feats)
                return super().__call__(noised_pos=noised_pos, t=t, feats=feats)

            @staticmethod
            def assert_shapes(y, t, feats):
                assert t.shape == (y.shape[0],)
                assert feats['atom_pad_mask'].shape == y.shape[:2]

        for beta in (0, 1000):
            kwargs = dict(num_timesteps=5, t_start=.1, tau=.02, beta=beta,
                          checkpoint_indices=(0, 2, 4), ess_threshold=1)
            score = lambda x, t: x[:, 0, 0]
            expected = self.run_sampler(FKSampler(**kwargs), noise=noise, score=score)
            sizes.clear()
            observed = self.run_sampler(FKSampler(model_batch_size=2, **kwargs),
                                        model=ChunkModel(), noise=noise, score=score)
            self.assertEqual(sizes, [2, 2, 1] * 5)
            self.assertEqual(observed['history'], expected['history'])
            torch.testing.assert_close(observed['denoised_coords'], expected['denoised_coords'],
                                       rtol=0, atol=0)
            if beta:
                # Parent 4 lies in a different model chunk from descendants 0/1.
                self.assertEqual(observed['history'][0]['parents'], [4] * 5)

    def test_shared_conditioning_expands_without_copying_feature_storage(self):
        source = {'coords': self.noise[:1].clone(),
                  'atom_pad_mask': self.batch['atom_pad_mask'][:1].clone(),
                  'dense_feature': torch.ones(1, 4, 4, dtype=torch.float64),
                  'scalar': torch.tensor(1), 'aa_seq': ['AAAA']}
        expanded = expand_shared_batch(source, 10)
        for key in ('coords', 'atom_pad_mask', 'dense_feature'):
            self.assertEqual(expanded[key].shape[0], 10)
            self.assertEqual(expanded[key].stride(0), 0)
            self.assertEqual(expanded[key].untyped_storage().data_ptr(),
                             source[key].untyped_storage().data_ptr())
        self.assertIs(expanded['scalar'], source['scalar'])
        self.assertIs(expanded['aa_seq'], source['aa_seq'])


if __name__ == "__main__":
    unittest.main()
