"""The val-time diagnostics that val_loss structurally cannot report.

val_loss scores the conditional MEAN velocity field at uniform t, averaged over
all action_horizon x action_dim entries. The deploy failure it misses is sampling
spread: independent draws from one observation disagreeing about the chunk end by
tenths of a metre. These tests pin the metric arithmetic against stub policies
with hand-computable answers.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from Manip_Flow.common.val_diagnostics import (  # noqa: E402
    ACTION_CHANNELS_PER_ARM,
    log_draw_dispersion,
    log_prefix_consistency,
    shuffled_obs_batch,
)

HORIZON = 4
ARMS = 2
ACTION_DIM = ARMS * ACTION_CHANNELS_PER_ARM


def _obs(batch: int) -> dict:
    # One low_dim key is enough: the metrics only slice and tile the batch axis.
    return {"robot0_eef_pos": torch.zeros(batch, 1, 3)}


class ScriptedDrawPolicy:
    """Returns a caller-supplied stack of draws, one per predict_action row."""

    def __init__(self, draws: torch.Tensor) -> None:
        # draws: (n_obs, k, Ta, action_dim), consumed in the tiled row order that
        # repeat_interleave produces.
        self._draws = draws

    def predict_action(self, obs_dict, *args, **kwargs):
        del obs_dict, args, kwargs
        n_obs, k = self._draws.shape[:2]
        return {"action_pred": self._draws.reshape(n_obs * k, HORIZON, ACTION_DIM)}


def test_draw_dispersion_reports_metres_of_position_spread() -> None:
    # Given: one observation, two draws that differ ONLY in arm 0's x position at
    # the last step -- by 0.4 m, the swing measured on the robot.
    draws = torch.zeros(1, 2, HORIZON, ACTION_DIM)
    draws[0, 1, -1, 0] = 0.4

    step_log: dict = {}
    log_draw_dispersion(step_log, ScriptedDrawPolicy(draws), _obs(1), n_obs=1, k=2)

    # Then: the last-step metric carries the spread in metres. torch.std is
    # unbiased, so two samples 0.4 apart have std 0.4/sqrt(2); the reported
    # last-step mean also averages over the two arms, and arm 1 is identical.
    expected_arm0 = 0.4 / math.sqrt(2.0)
    assert step_log["val_draw_std_pos_max_m"] == pytest.approx(expected_arm0, rel=1e-5)
    assert step_log["val_draw_std_pos_last_m"] == pytest.approx(
        expected_arm0 / ARMS, rel=1e-5
    )
    # And: rotation/width channels do not leak into the position metric.
    assert step_log["val_draw_std_width_last_m"] == pytest.approx(0.0)


def test_draw_dispersion_is_zero_for_a_deterministic_policy() -> None:
    # Given: every draw identical -- the "no sampling variance" reference.
    draws = torch.arange(2 * HORIZON * ACTION_DIM, dtype=torch.float32)
    draws = draws.reshape(2, 1, HORIZON, ACTION_DIM).repeat(1, 3, 1, 1)

    step_log: dict = {}
    log_draw_dispersion(step_log, ScriptedDrawPolicy(draws), _obs(2), n_obs=2, k=3)

    # Then: no spread, even though the actions themselves are large.
    for key in ("val_draw_std_pos_mean_m", "val_draw_std_pos_max_m",
                "val_draw_std_pos_last_m", "val_draw_std_width_last_m"):
        assert step_log[key] == pytest.approx(0.0, abs=1e-6)


def test_draw_dispersion_tiles_each_observation_k_times() -> None:
    # Given: a policy that records the batch it was asked to sample.
    class RecordingPolicy:
        def __init__(self) -> None:
            self.seen_rows = None

        def predict_action(self, obs_dict, *args, **kwargs):
            del args, kwargs
            rows = obs_dict["robot0_eef_pos"].shape[0]
            self.seen_rows = obs_dict["robot0_eef_pos"][:, 0, 0].tolist()
            return {"action_pred": torch.zeros(rows, HORIZON, ACTION_DIM)}

    obs = {"robot0_eef_pos": torch.tensor([[[0.0, 0, 0]], [[1.0, 0, 0]],
                                           [[2.0, 0, 0]]])}
    policy = RecordingPolicy()

    # When: only the first 2 of 3 observations are used, with k=3 draws each.
    log_draw_dispersion({}, policy, obs, n_obs=2, k=3)

    # Then: rows are grouped per observation (repeat_interleave, not repeat), so
    # the reshape to (n_obs, k, ...) groups draws of the SAME observation.
    assert policy.seen_rows == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


class TwoChunkPolicy:
    """First call returns `first`; the RTC-prefixed call returns `second`."""

    def __init__(self, first: torch.Tensor, second: torch.Tensor) -> None:
        self._chunks = [first, second]
        self.prefix_lengths: list = []

    def predict_action(self, obs_dict, *args, **kwargs):
        del obs_dict, args
        prefix = kwargs.get("rtc_action_prefix")
        self.prefix_lengths.append(None if prefix is None else prefix.shape[1])
        return {"action_pred": self._chunks.pop(0)}


def test_prefix_consistency_separates_guided_prefix_from_unguided_tail() -> None:
    # Given: a second draw that honours the committed prefix exactly but walks
    # 0.3 m away from it in arm 0's y channel over the unguided tail.
    first = torch.zeros(1, HORIZON, ACTION_DIM)
    second = first.clone()
    second[0, 2:, 1] = 0.3
    policy = TwoChunkPolicy(first, second)

    step_log: dict = {}
    log_prefix_consistency(step_log, policy, _obs(1), n_obs=1, prefix_steps=2)

    # Then: the guided region shows no disagreement...
    assert step_log["val_prefix_guided_pos_m"] == pytest.approx(0.0)
    # ...while the tail carries it, averaged over the two arms (arm 1 agrees).
    assert step_log["val_prefix_tail_pos_max_m"] == pytest.approx(0.3, rel=1e-5)
    assert step_log["val_prefix_tail_pos_mean_m"] == pytest.approx(0.3 / ARMS, rel=1e-5)
    assert step_log["val_prefix_tail_pos_last_m"] == pytest.approx(0.3 / ARMS, rel=1e-5)
    # And: the committed prefix really was handed to the RTC path.
    assert policy.prefix_lengths == [None, 2]


def test_prefix_consistency_flags_a_prefix_the_model_ignored() -> None:
    # Given: a second draw that violates the prefix it was guided toward.
    first = torch.zeros(1, HORIZON, ACTION_DIM)
    second = first.clone()
    second[0, :2, 0] = 0.5
    policy = TwoChunkPolicy(first, second)

    step_log: dict = {}
    log_prefix_consistency(step_log, policy, _obs(1), n_obs=1, prefix_steps=2)

    # Then: the guided metric is non-zero -- distinguishing "RTC did not bite"
    # from "the model is inconsistent about the future", which need different fixes.
    assert step_log["val_prefix_guided_pos_m"] == pytest.approx(0.5 / ARMS, rel=1e-5)
    assert step_log["val_prefix_tail_pos_mean_m"] == pytest.approx(0.0)


def test_prefix_consistency_keeps_one_unguided_step_when_prefix_overruns() -> None:
    # Given: a prefix request at/over the full horizon (rtc_execution_horizon can
    # exceed the sampled chunk in a debug/short-horizon config).
    first = torch.zeros(1, HORIZON, ACTION_DIM)
    second = first.clone()
    second[0, -1, 0] = 0.2
    policy = TwoChunkPolicy(first, second)

    step_log: dict = {}
    log_prefix_consistency(step_log, policy, _obs(1), n_obs=1, prefix_steps=99)

    # Then: it is clamped so the tail slice is never empty (an empty slice would
    # report nan and poison topk's metric dict).
    assert policy.prefix_lengths == [None, HORIZON - 1]
    assert not math.isnan(step_log["val_prefix_tail_pos_mean_m"])
    assert step_log["val_prefix_tail_pos_last_m"] == pytest.approx(0.2 / ARMS, rel=1e-5)


def test_shuffled_obs_baseline_breaks_pairing_but_keeps_the_action_marginal() -> None:
    # Given: a val batch whose actions are all distinct.
    batch = {
        "obs": {"robot0_eef_pos": torch.zeros(3, 1, 3)},
        "action": torch.arange(3 * HORIZON * ACTION_DIM, dtype=torch.float32).reshape(
            3, HORIZON, ACTION_DIM
        ),
    }

    shuffled = shuffled_obs_batch(batch)

    # Then: the observations are untouched, every action still appears exactly
    # once (the marginal is intact, so the loss floor is unchanged) and no row
    # keeps its own action -- which is what makes this an "uninformative
    # conditioning" reference rather than a harder version of the same task.
    assert shuffled["obs"] is batch["obs"]
    assert torch.equal(shuffled["action"].sort(dim=0).values,
                       batch["action"].sort(dim=0).values)
    assert not torch.equal(shuffled["action"], batch["action"])
    for row in range(3):
        assert not torch.equal(shuffled["action"][row], batch["action"][row])
