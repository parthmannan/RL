# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for single_controller_utils.utils pure helpers."""

from __future__ import annotations

import math

import pytest
import torch
from tensordict import TensorDict

from nemo_rl.algorithms.single_controller_utils.utils import (
    ImportanceSamplingDiagnosticsAccumulator,
    aggregate_step_metrics,
    apply_message_level_advantage_penalties,
    fields_for_put,
    reduce_advantage_pump_metrics,
    squeeze_trailing_unit_dim,
    tensor_field,
)
from nemo_rl.data_plane import KVBatchMeta


def _meta(size: int, sequence_lengths: list[int] | None = None) -> KVBatchMeta:
    return KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[f"s{i}" for i in range(size)],
        sequence_lengths=sequence_lengths,
    )


class TestSqueezeTrailingUnitDim:
    def test_squeezes_trailing_unit_dim(self) -> None:
        out = squeeze_trailing_unit_dim(torch.zeros(4, 1))
        assert out.shape == (4,)

    def test_leaves_1d_untouched(self) -> None:
        out = squeeze_trailing_unit_dim(torch.zeros(4))
        assert out.shape == (4,)

    def test_leaves_non_unit_trailing_dim(self) -> None:
        out = squeeze_trailing_unit_dim(torch.zeros(4, 3))
        assert out.shape == (4, 3)


class TestTensorField:
    def test_returns_dense_tensor(self) -> None:
        td = TensorDict({"x": torch.arange(6).reshape(2, 3)}, batch_size=[2])
        out = tensor_field(td, "x")
        assert torch.equal(out, torch.arange(6).reshape(2, 3))

    def test_pads_nested_tensor(self) -> None:
        nested = torch.nested.as_nested_tensor(
            [torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
            layout=torch.jagged,
        )
        td = TensorDict({"x": nested}, batch_size=[2])
        out = tensor_field(td, "x")
        assert not out.is_nested
        assert out.shape == (2, 3)
        assert out[1].tolist() == [4, 5, 0]

    def test_non_tensor_raises_type_error(self) -> None:
        td = TensorDict({"x": torch.zeros(2)}, batch_size=[2], non_blocking=False)
        td.set_non_tensor("meta", ["a", "b"])
        with pytest.raises(TypeError):
            tensor_field(td, "meta")


class TestAggregateStepMetrics:
    def test_scalar_loss_and_grad_norm_tensors(self) -> None:
        result = {
            "loss": torch.tensor([1.0, 3.0]),
            "grad_norm": torch.tensor(2.0),
        }
        out = aggregate_step_metrics(result)
        assert out["loss"] == pytest.approx(2.0)
        assert out["grad_norm"] == pytest.approx(2.0)

    def test_float_loss_and_optional_scalars(self) -> None:
        out = aggregate_step_metrics({"loss": 0.5, "total_flops": 10, "num_ranks": 4})
        assert out["loss"] == pytest.approx(0.5)
        assert out["total_flops"] == pytest.approx(10.0)
        assert out["num_ranks"] == 4
        assert "grad_norm" not in out

    def test_mb_metric_reduction_rules(self) -> None:
        result = {
            "all_mb_metrics": {
                "probs_ratio_min": [0.4, 0.2, 0.9],
                "probs_ratio_max": [0.4, 0.2, 0.9],
                "lr": [0.1, 0.3],
                "some_sum_metric": [1.0, 2.0, 3.0],
            }
        }
        out = aggregate_step_metrics(result)
        assert out["probs_ratio_min"] == pytest.approx(0.2)
        assert out["probs_ratio_max"] == pytest.approx(0.9)
        assert out["lr"] == pytest.approx(0.2)
        assert out["some_sum_metric"] == pytest.approx(6.0)

    def test_min_max_all_inf_falls_back_to_neg_one(self) -> None:
        result = {
            "all_mb_metrics": {
                "probs_ratio_min": [math.inf, math.inf],
                "probs_ratio_max": [math.inf],
            }
        }
        out = aggregate_step_metrics(result)
        assert out["probs_ratio_min"] == -1.0
        assert out["probs_ratio_max"] == -1.0

    def test_moe_and_mtp_metrics_are_prefixed(self) -> None:
        result = {
            "moe_metrics": {"load_balance": [1.0, 3.0]},
            "mtp_metrics": {"acc": [2.0, 2.0]},
        }
        out = aggregate_step_metrics(result)
        assert out["moe/load_balance"] == pytest.approx(4.0)
        assert out["mtp/acc"] == pytest.approx(4.0)


class TestReduceAdvantagePumpMetrics:
    def test_reward_and_advantages_and_tokens(self) -> None:
        out = reduce_advantage_pump_metrics(
            rewards=[torch.tensor([1.0, 3.0])],
            masked_advantages=[torch.tensor([-1.0, 0.0, 2.0])],
            sequence_lengths=[4, 6],
            num_mask_sample_filtered=[1, 2],
        )
        assert out["reward"] == pytest.approx(2.0)
        assert out["advantages/mean"] == pytest.approx(1.0 / 3.0)
        assert out["advantages/max"] == pytest.approx(2.0)
        assert out["advantages/min"] == pytest.approx(-1.0)
        assert out["total_num_tokens"] == pytest.approx(10.0)
        assert out["num_mask_sample_filtered"] == pytest.approx(3.0)

    def test_staleness_reduces_to_mean_min_max(self) -> None:
        out = reduce_advantage_pump_metrics(
            rewards=[],
            masked_advantages=[],
            sequence_lengths=[],
            stalenesses=[0, 1, 1, 2],
        )
        assert out["staleness/mean"] == pytest.approx(1.0)
        assert out["staleness/min"] == pytest.approx(0.0)
        assert out["staleness/max"] == pytest.approx(2.0)

    def test_staleness_omitted_when_absent(self) -> None:
        out = reduce_advantage_pump_metrics(
            rewards=[],
            masked_advantages=[],
            sequence_lengths=[],
        )
        assert not any(key.startswith("staleness/") for key in out)

    def test_empty_advantages_tensor_yields_zeros(self) -> None:
        out = reduce_advantage_pump_metrics(
            rewards=[],
            masked_advantages=[torch.empty(0)],
            sequence_lengths=[],
        )
        assert out["advantages/mean"] == 0.0
        assert out["advantages/max"] == 0.0
        assert out["advantages/min"] == 0.0
        assert "reward" not in out
        assert "total_num_tokens" not in out

    def test_all_empty_inputs_returns_empty_dict(self) -> None:
        assert reduce_advantage_pump_metrics([], [], []) == {}

    def test_seq_logprob_error_metrics_are_reduced_across_streaming_chunks(
        self,
    ) -> None:
        out = reduce_advantage_pump_metrics(
            rewards=[],
            masked_advantages=[],
            sequence_lengths=[],
            seq_logprob_error_metrics=[
                {
                    "max_seq_mult_prob_error": 10.0,
                    "mean_seq_mult_prob_error": 3.0,
                    "min_seq_mult_prob_error": 1.1,
                    "max_seq_mult_prob_error_after_mask": 1.5,
                    "mean_seq_mult_prob_error_after_mask": 1.2,
                    "min_seq_mult_prob_error_after_mask": 1.0,
                    "num_masked_seqs_by_logprob_error": 1,
                    "masked_correct_pct": 1.0,
                    "_num_valid_seqs_before": 4,
                    "_num_valid_seqs_after": 3,
                },
                {
                    "max_seq_mult_prob_error": 5.0,
                    "mean_seq_mult_prob_error": 2.0,
                    "min_seq_mult_prob_error": 1.05,
                    "max_seq_mult_prob_error_after_mask": 1.8,
                    "mean_seq_mult_prob_error_after_mask": 1.4,
                    "min_seq_mult_prob_error_after_mask": 1.02,
                    "num_masked_seqs_by_logprob_error": 2,
                    "masked_correct_pct": 0.0,
                    "_num_valid_seqs_before": 6,
                    "_num_valid_seqs_after": 4,
                },
            ],
        )

        assert out["max_seq_mult_prob_error"] == pytest.approx(10.0)
        assert out["mean_seq_mult_prob_error"] == pytest.approx(2.4)
        assert out["min_seq_mult_prob_error"] == pytest.approx(1.05)
        assert out["max_seq_mult_prob_error_after_mask"] == pytest.approx(1.8)
        assert out["mean_seq_mult_prob_error_after_mask"] == pytest.approx(9.2 / 7)
        assert out["min_seq_mult_prob_error_after_mask"] == pytest.approx(1.0)
        assert out["num_masked_seqs_by_logprob_error"] == 3
        assert out["masked_correct_pct"] == pytest.approx(1.0 / 3)

    def test_violation_rates_from_per_sample_counts(self) -> None:
        out = reduce_advantage_pump_metrics(
            rewards=[],
            masked_advantages=[],
            sequence_lengths=[],
            num_invalid_tool_calls=[1, 0, 1],
            num_malformed_thinking=[0, 1, 0],
            num_assistant_messages=[2, 1, 1],
        )
        assert out["invalid_tool_call_rate"] == pytest.approx(0.5)
        assert out["malformed_thinking_rate"] == pytest.approx(0.25)
        assert out["num_invalid_tool_calls"] == pytest.approx(2.0)
        assert out["num_malformed_thinking"] == pytest.approx(1.0)
        assert out["num_assistant_messages"] == pytest.approx(4.0)

    def test_no_assistant_messages_omits_violation_metrics(self) -> None:
        assert (
            reduce_advantage_pump_metrics(
                [],
                [],
                [],
                num_invalid_tool_calls=[0],
                num_malformed_thinking=[0],
                num_assistant_messages=[0],
            )
            == {}
        )


class TestApplyMessageLevelAdvantagePenalties:
    def test_overwrites_only_flagged_tokens(self) -> None:
        advantages = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = apply_message_level_advantage_penalties(
            advantages,
            invalid_tool_call_mask=torch.tensor([[False, True, False, False]]),
            malformed_thinking_mask=torch.tensor([[False, False, True, False]]),
            invalid_tool_call_advantage=-5.0,
            malformed_thinking_advantage=-7.0,
        )
        torch.testing.assert_close(result, torch.tensor([[1.0, -5.0, -7.0, 4.0]]))
        torch.testing.assert_close(advantages, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    def test_invalid_tool_call_takes_precedence_on_overlap(self) -> None:
        result = apply_message_level_advantage_penalties(
            torch.zeros(1, 2),
            invalid_tool_call_mask=torch.tensor([[False, True]]),
            malformed_thinking_mask=torch.tensor([[False, True]]),
            invalid_tool_call_advantage=-5.0,
            malformed_thinking_advantage=-7.0,
        )
        torch.testing.assert_close(result, torch.tensor([[0.0, -5.0]]))

    def test_only_invalid_tool_call_penalty_leaves_malformed_untouched(
        self,
    ) -> None:
        result = apply_message_level_advantage_penalties(
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            invalid_tool_call_mask=torch.tensor([[False, True, False, False]]),
            malformed_thinking_mask=torch.tensor([[False, False, True, False]]),
            invalid_tool_call_advantage=-5.0,
            malformed_thinking_advantage=None,
        )
        torch.testing.assert_close(result, torch.tensor([[1.0, -5.0, 3.0, 4.0]]))

    def test_only_malformed_thinking_penalty_leaves_invalid_untouched(
        self,
    ) -> None:
        result = apply_message_level_advantage_penalties(
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            invalid_tool_call_mask=torch.tensor([[False, True, False, False]]),
            malformed_thinking_mask=torch.tensor([[False, False, True, False]]),
            invalid_tool_call_advantage=None,
            malformed_thinking_advantage=-7.0,
        )
        torch.testing.assert_close(result, torch.tensor([[1.0, 2.0, -7.0, 4.0]]))

    def test_disabled_penalties_leave_advantages_unchanged(self) -> None:
        advantages = torch.tensor([[1.0, 2.0]])
        result = apply_message_level_advantage_penalties(
            advantages,
            invalid_tool_call_mask=torch.tensor([[True, False]]),
            malformed_thinking_mask=torch.tensor([[False, True]]),
            invalid_tool_call_advantage=None,
            malformed_thinking_advantage=None,
        )
        assert result is advantages

    def test_rejects_misaligned_mask(self) -> None:
        with pytest.raises(ValueError, match="invalid_tool_call_mask shape"):
            apply_message_level_advantage_penalties(
                torch.zeros(1, 2),
                invalid_tool_call_mask=torch.zeros(1, 3, dtype=torch.bool),
                malformed_thinking_mask=torch.zeros(1, 2, dtype=torch.bool),
                invalid_tool_call_advantage=-5.0,
                malformed_thinking_advantage=None,
            )


class TestFieldsForPut:
    def test_no_sequence_lengths_packs_contiguous(self) -> None:
        meta = _meta(2, sequence_lengths=None)
        out = fields_for_put(meta, {"advantages": torch.zeros(2, 3)})
        assert out.batch_size == torch.Size([2])
        assert not out["advantages"].is_nested
        assert out["advantages"].shape == (2, 3)

    def test_renests_padded_rows_by_sequence_length(self) -> None:
        meta = _meta(2, sequence_lengths=[3, 2])
        value = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]])
        out = fields_for_put(meta, {"advantages": value})
        assert out["advantages"].is_nested
        rows = out["advantages"].unbind()
        assert rows[0].tolist() == [1.0, 2.0, 3.0]
        assert rows[1].tolist() == [4.0, 5.0]

    def test_non_matching_width_stays_contiguous(self) -> None:
        meta = _meta(2, sequence_lengths=[3, 2])
        value = torch.zeros(2, 1)
        out = fields_for_put(meta, {"scalar": value})
        assert not out["scalar"].is_nested
        assert out["scalar"].shape == (2, 1)


def _is_accumulator() -> ImportanceSamplingDiagnosticsAccumulator:
    return ImportanceSamplingDiagnosticsAccumulator(
        sequence_level_importance_ratios=False,
        truncated_importance_sampling_ratio=2.0,
        truncated_importance_sampling_ratio_min=0.5,
        truncated_importance_sampling_type=None,
    )


def _record_batch(
    acc: ImportanceSamplingDiagnosticsAccumulator,
    *,
    environments: list[str],
    weight_versions: list[int],
    trainer_version: int = 5,
    errors: list[float] | None = None,
    sample_mask: list[float] | None = None,
) -> None:
    size = len(environments)
    seq_error = torch.tensor(errors or [1.2] * size)
    acc.record(
        step=1,
        trainer_version=trainer_version,
        sample_ids=[f"grp{i}_g0" for i in range(size)],
        rollout_tags=[{"rollout_environment": env} for env in environments],
        rollout_weight_versions=weight_versions,
        sequence_lengths=[8] * size,
        prev_logprobs=torch.zeros(size, 4),
        generation_logprobs=torch.zeros(size, 4),
        token_mask=torch.ones(size, 4),
        sample_mask=torch.tensor(sample_mask or [1.0] * size),
        advantages=torch.zeros(size, 4),
        rewards=torch.ones(size),
        seq_mult_prob_error=seq_error,
        valid_seq_mask=torch.ones(size, dtype=torch.bool),
    )


class TestImportanceSamplingDiagnosticsAccumulator:
    def test_empty_flush_yields_nothing(self) -> None:
        assert _is_accumulator().flush() == ({}, [])

    def test_splits_ifbench_cohort_and_buckets_by_lag(self) -> None:
        acc = _is_accumulator()
        # Two lags (5-5=0 and 5-4=1); within lag 0, one ifbench row and one not.
        _record_batch(
            acc,
            environments=[
                "instruction_following_simple_agent",
                "math",
                "math",
            ],
            weight_versions=[5, 5, 4],
            errors=[1.4, 1.05, 1.3],
        )
        metrics, rows = acc.flush()

        assert metrics["importance_sampling/lag_0/num_sequences"] == 2.0
        assert metrics["importance_sampling/lag_1/num_sequences"] == 1.0
        # Only the ifbench row lands in the lag-0 ifbench cohort.
        assert metrics["importance_sampling/lag_0/ifbench_direct/num_sequences"] == 1.0
        assert metrics["importance_sampling/lag_0/ifbench_direct/retained_ei_mean"] == (
            pytest.approx(1.4)
        )
        assert metrics["importance_sampling/lag_1/ifbench_direct/num_sequences"] == 0.0
        assert metrics["importance_sampling/lag_0/retained_ei_mean"] == pytest.approx(
            (1.4 + 1.05) / 2
        )
        # One lag_summary per lag, plus the retained high-error rows.
        summaries = [r for r in rows if r["record_type"] == "lag_summary"]
        assert [s["observed_lag"] for s in summaries] == [0, 1]
        assert summaries[0]["all"]["num_sequences"] == 2
        assert summaries[0]["ifbench_direct"]["num_sequences"] == 1
        assert summaries[0]["other"]["num_sequences"] == 1
        high = [r for r in rows if r["record_type"] == "high_ei_sequence"]
        assert {r["is_ifbench_direct"] for r in high} == {True, False}

    def test_masked_rows_count_but_are_not_retained(self) -> None:
        acc = _is_accumulator()
        _record_batch(
            acc,
            environments=["math", "math"],
            weight_versions=[5, 5],
            errors=[1.1, 9.0],
            sample_mask=[1.0, 0.0],
        )
        metrics, _ = acc.flush()
        assert metrics["importance_sampling/lag_0/num_sequences"] == 2.0
        assert metrics["importance_sampling/lag_0/masked_sequence_fraction"] == (
            pytest.approx(0.5)
        )
        # The masked row's error must not drag the retained mean.
        assert metrics["importance_sampling/lag_0/retained_ei_mean"] == pytest.approx(
            1.1
        )

    def test_flush_resets_state(self) -> None:
        acc = _is_accumulator()
        _record_batch(acc, environments=["math"], weight_versions=[5])
        acc.flush()
        assert acc.flush() == ({}, [])

    def test_mixing_optimizer_steps_is_rejected(self) -> None:
        acc = _is_accumulator()
        _record_batch(acc, environments=["math"], weight_versions=[5])
        with pytest.raises(ValueError, match="mixed optimizer steps"):
            acc.record(
                step=2,
                trainer_version=5,
                sample_ids=["grp0_g0"],
                rollout_tags=[{"rollout_environment": "math"}],
                rollout_weight_versions=[5],
                sequence_lengths=[8],
                prev_logprobs=torch.zeros(1, 4),
                generation_logprobs=torch.zeros(1, 4),
                token_mask=torch.ones(1, 4),
                sample_mask=torch.ones(1),
                advantages=torch.zeros(1, 4),
                rewards=torch.ones(1),
                seq_mult_prob_error=torch.tensor([1.2]),
                valid_seq_mask=torch.ones(1, dtype=torch.bool),
            )

    def test_misaligned_batch_metadata_is_rejected(self) -> None:
        acc = _is_accumulator()
        with pytest.raises(ValueError, match="misaligned"):
            acc.record(
                step=1,
                trainer_version=5,
                sample_ids=["grp0_g0"],
                rollout_tags=[{"rollout_environment": "math"}],
                rollout_weight_versions=[5],
                sequence_lengths=[8],
                prev_logprobs=torch.zeros(2, 4),
                generation_logprobs=torch.zeros(2, 4),
                token_mask=torch.ones(2, 4),
                sample_mask=torch.ones(2),
                advantages=torch.zeros(2, 4),
                rewards=torch.ones(2),
                seq_mult_prob_error=torch.tensor([1.2, 1.3]),
                valid_seq_mask=torch.ones(2, dtype=torch.bool),
            )
