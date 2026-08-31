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

"""Helpers used by SingleControllerActor."""

from __future__ import annotations

import hashlib
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.experience.interfaces import (
    NEMO_GYM_RESERVED_KEY_PREFIX,
    ROLLOUT_ENV_EXTRA_TAG_PREFIX,
    ROLLOUT_ENVIRONMENT_TAG,
    ROLLOUT_GENERATION_LENGTH_TAG,
    ROLLOUT_REWARD_TAG,
    ROLLOUT_TRUNCATED_TAG,
)

# Reduction rules for all_mb_metrics. Mirror grpo.py / grpo_sync.py.
_MB_METRIC_MIN: frozenset[str] = frozenset(
    {"probs_ratio_min", "probs_ratio_clamped_min"}
)
_MB_METRIC_MAX: frozenset[str] = frozenset(
    {"probs_ratio_max", "probs_ratio_clamped_max"}
)
_MB_METRIC_MEAN: frozenset[str] = frozenset(
    {
        "lr",
        "wd",
        "reward",
        "global_valid_seqs",
        "global_valid_toks",
        "mean_prompt_length",
    }
)
_METRIC_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def _rollout_environment_metric_component(environment: str) -> str:
    """Return a readable metric component without silent name collisions."""
    sanitized = _METRIC_COMPONENT_PATTERN.sub("_", environment).strip("_.")
    if sanitized == environment:
        return sanitized
    digest = hashlib.blake2s(environment.encode(), digest_size=8).hexdigest()
    return f"{sanitized or 'unknown'}-{digest}"


def _scalar_summary(
    values: list[float],
    prefix: str,
    *,
    mean_denominator: int | None = None,
) -> dict[str, float]:
    """Return the scalar portion of the legacy rollout metric family.

    V1 divides sparse ``env_extras`` sums by the full environment cohort, not
    by the number of samples that supplied that specific key. The optional
    denominator preserves that behavior without carrying W&B Histogram objects
    through checkpointed SingleController state.
    """
    denominator = mean_denominator if mean_denominator is not None else len(values)
    return {
        f"{prefix}/mean": sum(values) / denominator,
        f"{prefix}/max": max(values),
        f"{prefix}/min": min(values),
        f"{prefix}/median": statistics.median(values),
        f"{prefix}/stddev": statistics.stdev(values)
        if len(values) > 1
        else math.nan,
    }

_MAX_IMPORTANCE_DIAGNOSTIC_SEQUENCES_PER_LAG = 16


@dataclass
class _ImportanceSamplingBucket:
    """Sufficient statistics for sampled sequences in one lag bucket."""

    num_sequences: int = 0
    num_tokens: int = 0
    num_nonfinite_tokens: int = 0
    log_ratio_sum: float = 0.0
    abs_log_ratio_sum: float = 0.0
    tis_oob_count: int = 0
    objective_signal_sum: float = 0.0

    def update(
        self,
        *,
        finite_token_count: int,
        nonfinite_token_count: int,
        log_ratio_sum: float,
        abs_log_ratio_sum: float,
        tis_oob_count: int,
        objective_signal_sum: float,
    ) -> None:
        self.num_sequences += 1
        self.num_tokens += finite_token_count
        self.num_nonfinite_tokens += nonfinite_token_count
        self.log_ratio_sum += log_ratio_sum
        self.abs_log_ratio_sum += abs_log_ratio_sum
        self.tis_oob_count += tis_oob_count
        self.objective_signal_sum += objective_signal_sum

    def mean(self, total: float) -> float:
        return total / self.num_tokens if self.num_tokens else 0.0


@dataclass
class ImportanceSamplingDiagnosticsAccumulator:
    """Accumulate raw actor/behavior log-ratio diagnostics for one train step."""

    use_importance_sampling_correction: bool
    sequence_level_importance_ratios: bool
    truncated_importance_sampling_ratio: float | None
    truncated_importance_sampling_ratio_min: float | None
    truncated_importance_sampling_type: str | None
    _rows: list[dict[str, Any]] = field(default_factory=list, init=False)
    _all_bucket: _ImportanceSamplingBucket = field(
        default_factory=_ImportanceSamplingBucket, init=False
    )
    _buckets_by_lag: dict[int, _ImportanceSamplingBucket] = field(
        default_factory=dict, init=False
    )
    _population_sequences_by_lag: dict[int, int] = field(
        default_factory=dict, init=False
    )

    def _post_tis_weights(
        self, log_ratios: torch.Tensor, finite_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        upper = self.truncated_importance_sampling_ratio
        lower = self.truncated_importance_sampling_ratio_min
        tis_type = self.truncated_importance_sampling_type
        finite_log_ratios = torch.where(finite_mask, log_ratios, 0.0)
        raw_token_weights = torch.nan_to_num(
            torch.exp(finite_log_ratios), nan=0.0, posinf=0.0, neginf=0.0
        )

        def outside_bounds(value: torch.Tensor) -> torch.Tensor:
            result = torch.zeros_like(value, dtype=torch.bool)
            if upper is not None:
                result |= value > math.log(upper)
            if lower is not None and lower > 0:
                result |= value < math.log(lower)
            return result

        if self.sequence_level_importance_ratios:
            sequence_log_ratio = finite_log_ratios.sum(dim=1, keepdim=True)
            sequence_weight = torch.nan_to_num(
                torch.exp(sequence_log_ratio), nan=0.0, posinf=0.0, neginf=0.0
            )
            raw_weights = sequence_weight.expand_as(raw_token_weights)
            oob = outside_bounds(sequence_log_ratio).expand_as(log_ratios)
        elif tis_type == "seq-mask-tis":
            sequence_mean_log_ratio = finite_log_ratios.sum(
                dim=1, keepdim=True
            ) / finite_mask.sum(dim=1, keepdim=True).clamp_min(1)
            raw_weights = raw_token_weights
            oob = outside_bounds(sequence_mean_log_ratio).expand_as(log_ratios)
        else:
            raw_weights = raw_token_weights
            oob = torch.zeros_like(log_ratios, dtype=torch.bool)
            if upper is not None:
                oob |= log_ratios > math.log(upper)
            if lower is not None and lower > 0:
                oob |= log_ratios < math.log(lower)
        oob = oob & finite_mask

        if tis_type is None:
            return raw_weights, oob
        if tis_type == "tis":
            assert upper is not None
            return raw_weights.clamp(min=lower or 0.0, max=upper), oob
        if tis_type == "icepop":
            return torch.where(oob, torch.zeros_like(raw_weights), raw_weights), oob
        if tis_type == "seq-mask-tis":
            sequence_oob = oob.any(dim=1, keepdim=True)
            return (
                torch.where(sequence_oob, torch.zeros_like(raw_weights), raw_weights),
                oob,
            )
        raise ValueError(f"unsupported truncated importance sampling type: {tis_type}")

    def record(
        self,
        *,
        step: int,
        trainer_version: int,
        sample_ids: list[str],
        rollout_weight_versions: list[int],
        sequence_lengths: list[int] | None,
        prev_logprobs: torch.Tensor,
        generation_logprobs: torch.Tensor,
        token_mask: torch.Tensor,
        sample_mask: torch.Tensor,
        advantages: torch.Tensor,
        rewards: torch.Tensor,
    ) -> None:
        """Record one selected sampler tranche before its optimizer update."""
        batch_size = prev_logprobs.shape[0]
        if not (
            len(sample_ids) == len(rollout_weight_versions) == batch_size
            and rewards.numel() == batch_size
            and (sequence_lengths is None or len(sequence_lengths) == batch_size)
            and all(
                tensor.shape[0] == batch_size
                for tensor in (
                    prev_logprobs,
                    generation_logprobs,
                    token_mask,
                    sample_mask,
                    advantages,
                )
            )
        ):
            raise ValueError(
                "importance-sampling diagnostic batch metadata is misaligned"
            )

        selected_indices: list[int] = []
        selected_lags: list[int] = []
        selected_this_call: dict[int, int] = {}
        for i, rollout_version in enumerate(rollout_weight_versions):
            lag = int(trainer_version - int(rollout_version))
            self._population_sequences_by_lag[lag] = (
                self._population_sequences_by_lag.get(lag, 0) + 1
            )
            bucket = self._buckets_by_lag.get(lag)
            already_selected = bucket.num_sequences if bucket is not None else 0
            if (
                already_selected + selected_this_call.get(lag, 0)
                < _MAX_IMPORTANCE_DIAGNOSTIC_SEQUENCES_PER_LAG
            ):
                selected_indices.append(i)
                selected_lags.append(lag)
                selected_this_call[lag] = selected_this_call.get(lag, 0) + 1

        if not selected_indices:
            return

        selected = torch.tensor(selected_indices, device=prev_logprobs.device)
        log_ratios = (
            prev_logprobs.index_select(0, selected)[:, 1:]
            - generation_logprobs.index_select(0, selected)[:, 1:]
        ).float()
        valid_mask = token_mask.index_select(0, selected)[:, 1:].bool()
        valid_mask &= sample_mask.index_select(0, selected).bool().unsqueeze(-1)
        finite_mask = valid_mask & torch.isfinite(log_ratios)
        finite_log_ratios = torch.where(finite_mask, log_ratios, 0.0)
        token_advantages = advantages.index_select(0, selected)[:, 1:].float()
        post_tis_weights, oob_mask = self._post_tis_weights(log_ratios, finite_mask)
        objective_weights = (
            post_tis_weights
            if self.use_importance_sampling_correction
            else torch.ones_like(post_tis_weights)
        )
        objective_signal = torch.where(
            finite_mask, token_advantages.abs() * objective_weights, 0.0
        )
        response_token_counts = valid_mask.sum(dim=1)
        finite_token_counts = finite_mask.sum(dim=1)
        stats = (
            torch.stack(
                (
                    response_token_counts,
                    finite_token_counts,
                    finite_log_ratios.sum(dim=1),
                    finite_log_ratios.abs().sum(dim=1),
                    oob_mask.sum(dim=1),
                    objective_signal.sum(dim=1),
                    (token_advantages.ne(0) & finite_mask).sum(dim=1),
                    rewards.flatten().index_select(0, selected),
                ),
                dim=1,
            )
            .detach()
            .cpu()
        )

        for row_index, (i, lag) in enumerate(zip(selected_indices, selected_lags)):
            (
                response_token_count_value,
                finite_token_count_value,
                log_ratio_sum_value,
                abs_log_ratio_sum_value,
                tis_oob_count_value,
                objective_signal_sum_value,
                nonzero_advantage_count_value,
                reward_value,
            ) = stats[row_index].tolist()
            response_token_count = int(response_token_count_value)
            finite_token_count = int(finite_token_count_value)
            nonfinite_token_count = response_token_count - finite_token_count
            log_ratio_sum = float(log_ratio_sum_value)
            abs_log_ratio_sum = float(abs_log_ratio_sum_value)
            tis_oob_count = int(tis_oob_count_value)
            objective_signal_sum = float(objective_signal_sum_value)
            denominator = finite_token_count or 1
            sequence_mean_log_ratio = log_ratio_sum / denominator
            token_abs_log_ratio_mean = abs_log_ratio_sum / denominator
            tis_oob_fraction = tis_oob_count / denominator
            nonzero_advantage_fraction = (
                float(nonzero_advantage_count_value) / denominator
            )
            objective_signal_proxy_mean = objective_signal_sum / denominator

            for bucket in (
                self._all_bucket,
                self._buckets_by_lag.setdefault(lag, _ImportanceSamplingBucket()),
            ):
                bucket.update(
                    finite_token_count=finite_token_count,
                    nonfinite_token_count=nonfinite_token_count,
                    log_ratio_sum=log_ratio_sum,
                    abs_log_ratio_sum=abs_log_ratio_sum,
                    tis_oob_count=tis_oob_count,
                    objective_signal_sum=objective_signal_sum,
                )

            self._rows.append(
                {
                    "step": int(step),
                    "sample_id": sample_ids[i],
                    "observed_lag": lag,
                    "total_sequence_length": (
                        int(sequence_lengths[i])
                        if sequence_lengths is not None
                        else None
                    ),
                    "response_token_count": response_token_count,
                    "nonfinite_log_ratio_token_count": nonfinite_token_count,
                    "reward": float(reward_value),
                    "raw_sequence_mean_log_ratio": sequence_mean_log_ratio,
                    "raw_token_abs_log_ratio_mean": token_abs_log_ratio_mean,
                    "tis_oob_fraction": tis_oob_fraction,
                    "nonzero_advantage_fraction": nonzero_advantage_fraction,
                    "objective_signal_proxy_mean": objective_signal_proxy_mean,
                }
            )

    @staticmethod
    def _summarize_bucket(
        *, label: str, bucket: _ImportanceSamplingBucket, population: int
    ) -> dict[str, float]:
        prefix = f"importance_sampling/{label}"
        return {
            f"{prefix}/num_sequences": float(bucket.num_sequences),
            f"{prefix}/population_num_sequences": float(population),
            f"{prefix}/sequence_sample_fraction": (
                bucket.num_sequences / population if population else 0.0
            ),
            f"{prefix}/num_tokens": float(bucket.num_tokens),
            f"{prefix}/num_nonfinite_tokens": float(bucket.num_nonfinite_tokens),
            f"{prefix}/raw_token_log_ratio_mean": bucket.mean(bucket.log_ratio_sum),
            f"{prefix}/raw_token_abs_log_ratio_mean": bucket.mean(
                bucket.abs_log_ratio_sum
            ),
            f"{prefix}/tis_oob_fraction": bucket.mean(float(bucket.tis_oob_count)),
            f"{prefix}/objective_signal_proxy_mean": bucket.mean(
                bucket.objective_signal_sum
            ),
        }

    def flush(self) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Return step metrics and rows, then reset the accumulator."""
        if not self._rows:
            return {}, []

        metrics = self._summarize_bucket(
            label="all",
            bucket=self._all_bucket,
            population=sum(self._population_sequences_by_lag.values()),
        )
        for lag, bucket in sorted(self._buckets_by_lag.items()):
            metrics.update(
                self._summarize_bucket(
                    label=f"lag_{lag}",
                    bucket=bucket,
                    population=self._population_sequences_by_lag[lag],
                )
            )

        rows = self._rows
        self._rows = []
        self._all_bucket = _ImportanceSamplingBucket()
        self._buckets_by_lag = {}
        self._population_sequences_by_lag = {}
        return metrics, rows


def aggregate_step_metrics(train_result: dict[str, Any]) -> dict[str, Any]:
    """Reduce per-microbatch metric lists into step-level scalars.

    Args:
        train_result: Output of TQPolicy.finish_train_step.

    Returns:
        Flat dict of step-level scalars ready for logging.
    """
    metrics: dict[str, Any] = {}
    loss = train_result.get("loss")
    if isinstance(loss, torch.Tensor):
        metrics["loss"] = loss.detach().mean().item()
    elif loss is not None:
        metrics["loss"] = float(loss)
    grad_norm = train_result.get("grad_norm")
    if isinstance(grad_norm, torch.Tensor):
        metrics["grad_norm"] = grad_norm.detach().mean().item()
    elif grad_norm is not None:
        metrics["grad_norm"] = float(grad_norm)
    if "total_flops" in train_result:
        metrics["total_flops"] = float(train_result["total_flops"])
    if "num_ranks" in train_result:
        metrics["num_ranks"] = int(train_result["num_ranks"])

    # moe/mtp share the same reduction rules as all_mb_metrics in grpo.py.
    mb: dict[str, list[Any]] = {}
    if "moe_metrics" in train_result:
        mb.update({f"moe/{k}": v for k, v in train_result["moe_metrics"].items()})
    if "mtp_metrics" in train_result:
        mb.update({f"mtp/{k}": v for k, v in train_result["mtp_metrics"].items()})
    mb.update(train_result.get("all_mb_metrics", {}))

    for k, v in mb.items():
        if k in _MB_METRIC_MIN:
            valid = [x for x in v if not np.isinf(x)]
            metrics[k] = float(np.min(valid)) if valid else -1.0
        elif k in _MB_METRIC_MAX:
            valid = [x for x in v if not np.isinf(x)]
            metrics[k] = float(np.max(valid)) if valid else -1.0
        elif k in _MB_METRIC_MEAN:
            metrics[k] = float(np.mean(v))
        else:
            metrics[k] = float(np.sum(v))
    return metrics


def reduce_advantage_pump_metrics(
    rewards: list[torch.Tensor],
    masked_advantages: list[torch.Tensor],
    sequence_lengths: list[int],
    seq_logprob_error_metrics: list[dict[str, float]] | None = None,
    pass_rates: list[float] | None = None,
    stalenesses: list[int] | None = None,
    intended_pass_rates: list[float] | None = None,
) -> dict[str, float]:
    """Reduce per-step accumulators from _advantage_stage into step scalars.

    Args:
        rewards: One tensor per advantage_stage call; each row a sample reward.
        masked_advantages: Token-masked advantages, one tensor per call.
        sequence_lengths: All input_lengths trained on this step.
        seq_logprob_error_metrics: Sequence-error metrics and their aggregation
            counts, one record per streaming chunk.
        pass_rates: Per-dispatched-sample dataset pass_rate values accumulated
            over the training step (may span multiple sampler dispatches).
        stalenesses: Per-dispatched-sample staleness values (end_weight -
            start_weight at commit time). Same value across the N rows of one
            group, so mean-over-rows equals mean-over-groups.
        intended_pass_rates: Per-prompt dataset pass_rate values collected by
            _rollout_pump for every loader batch admitted since the last
            optimizer step. Reflects what the dataloader intended to feed;
            compare against pass_rate/* (what the trainer actually consumed)
            to spot sampler-driven difficulty skew.

    Returns:
        Step-level reward, advantage, token-count, optional sequence
        log-probability error metrics,
        pass_rate/{mean,std,min,max,num_samples} when pass_rates is non-empty,
        staleness/{mean,min,max} when stalenesses is non-empty, and
        intended_pass_rate/{mean,min,max,num_prompts} when intended_pass_rates
        is non-empty.
    """
    out: dict[str, float] = {}
    if rewards:
        reward_values = (
            torch.cat([reward.detach().flatten().cpu() for reward in rewards])
            .tolist()
        )
        if reward_values:
            out["reward"] = statistics.mean(reward_values)
            out.update(_scalar_summary(reward_values, "total_reward"))
    if masked_advantages:
        cat = torch.cat([a.flatten() for a in masked_advantages])
        if cat.numel() > 0:
            out["advantages/mean"] = float(cat.mean())
            out["advantages/max"] = float(cat.max())
            out["advantages/min"] = float(cat.min())
        else:
            out["advantages/mean"] = 0.0
            out["advantages/max"] = 0.0
            out["advantages/min"] = 0.0
    if sequence_lengths:
        out["total_num_tokens"] = float(sum(sequence_lengths))
    if seq_logprob_error_metrics:
        out.update(_reduce_seq_logprob_error_metrics(seq_logprob_error_metrics))
    if pass_rates:
        arr = np.asarray(pass_rates, dtype=np.float64)
        out["pass_rate/mean"] = float(arr.mean())
        out["pass_rate/std"] = float(arr.std())
        out["pass_rate/min"] = float(arr.min())
        out["pass_rate/max"] = float(arr.max())
        out["pass_rate/num_samples"] = float(arr.size)
    if stalenesses:
        arr = np.asarray(stalenesses, dtype=np.float64)
        out["staleness/mean"] = float(arr.mean())
        out["staleness/min"] = float(arr.min())
        out["staleness/max"] = float(arr.max())
    if intended_pass_rates:
        arr = np.asarray(intended_pass_rates, dtype=np.float64)
        out["intended_pass_rate/mean"] = float(arr.mean())
        out["intended_pass_rate/min"] = float(arr.min())
        out["intended_pass_rate/max"] = float(arr.max())
        out["intended_pass_rate/num_prompts"] = float(arr.size)
    return out


def reduce_rollout_length_metrics(
    rollout_tags: list[dict[str, Any]],
) -> dict[str, float]:
    """Aggregate generated-token lengths and rewards by rollout environment.

    The tags come from the ``KVBatchMeta`` chunks selected for one optimizer
    step, so streaming completion order cannot shift a sample into the wrong
    step's metrics.

    Args:
        rollout_tags: Per-sample metadata tags accumulated across train chunks.

    Returns:
        Global mean generation length, per-environment length summaries, and
        legacy-style per-environment reward summaries when their respective tags
        cover every sample. Missing tags are reported separately for lengths and
        rewards; incomplete cohorts suppress only the affected summary family.
    """
    by_environment: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    rewards_by_environment: dict[str, list[float]] = defaultdict(list)
    extras_by_environment: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    environment_sample_counts: dict[str, int] = defaultdict(int)
    missing_length_samples = 0
    missing_reward_samples = 0
    for tag in rollout_tags:
        environment = tag.get(ROLLOUT_ENVIRONMENT_TAG)
        valid_environment = isinstance(environment, str) and bool(environment)
        if valid_environment:
            environment_sample_counts[environment] += 1
        generation_length = tag.get(ROLLOUT_GENERATION_LENGTH_TAG)
        truncated = tag.get(ROLLOUT_TRUNCATED_TAG)
        if (
            not valid_environment
            or isinstance(generation_length, bool)
            or not isinstance(generation_length, (int, float))
            or not isinstance(truncated, bool)
        ):
            missing_length_samples += 1
        else:
            generation_length = float(generation_length)
            if not math.isfinite(generation_length) or generation_length < 0:
                missing_length_samples += 1
            else:
                by_environment[environment].append((generation_length, truncated))

        reward = tag.get(ROLLOUT_REWARD_TAG)
        if (
            not valid_environment
            or isinstance(reward, bool)
            or not isinstance(reward, (int, float))
        ):
            missing_reward_samples += 1
        else:
            reward = float(reward)
            if not math.isfinite(reward):
                missing_reward_samples += 1
            else:
                rewards_by_environment[environment].append(reward)

        if valid_environment:
            for tag_key, value in tag.items():
                if not tag_key.startswith(ROLLOUT_ENV_EXTRA_TAG_PREFIX):
                    continue
                metric_key = tag_key.removeprefix(ROLLOUT_ENV_EXTRA_TAG_PREFIX)
                if (
                    not metric_key
                    or metric_key.startswith(NEMO_GYM_RESERVED_KEY_PREFIX)
                    or not isinstance(value, (bool, int, float))
                ):
                    continue
                numeric_value = float(value)
                if math.isfinite(numeric_value):
                    extras_by_environment[environment][metric_key].append(
                        numeric_value
                    )

    tagged_length_samples = sum(len(rows) for rows in by_environment.values())
    tagged_reward_samples = sum(
        len(values) for values in rewards_by_environment.values()
    )
    total_length_samples = tagged_length_samples + missing_length_samples
    total_reward_samples = tagged_reward_samples + missing_reward_samples
    metrics: dict[str, float] = {
        "rollout_length/tagged_samples": float(tagged_length_samples),
        "rollout_length/missing_samples": float(missing_length_samples),
        "rollout_length/tag_coverage": (
            tagged_length_samples / total_length_samples
            if total_length_samples
            else 0.0
        ),
        "rollout_reward/tagged_samples": float(tagged_reward_samples),
        "rollout_reward/missing_samples": float(missing_reward_samples),
        "rollout_reward/tag_coverage": (
            tagged_reward_samples / total_reward_samples
            if total_reward_samples
            else 0.0
        ),
    }
    # Older checkpoints predate these tags. Mixing their rows with newly
    # generated rows and reporting only the tagged subset would make the first
    # resumed step look complete while excluding exactly the restored samples
    # under investigation.
    if not missing_length_samples and tagged_length_samples:
        all_lengths: list[float] = []
        for environment in sorted(by_environment):
            rows = by_environment[environment]
            lengths = np.asarray([length for length, _ in rows], dtype=np.float64)
            all_lengths.extend(lengths.tolist())
            metric_environment = _rollout_environment_metric_component(environment)
            prefix = f"rollout_length/{metric_environment}"
            metrics.update(
                {
                    f"{prefix}/count": float(len(rows)),
                    f"{prefix}/mean": float(np.mean(lengths)),
                    f"{prefix}/stddev": float(np.std(lengths)),
                    f"{prefix}/min": float(np.min(lengths)),
                    f"{prefix}/p50": float(np.percentile(lengths, 50)),
                    f"{prefix}/p95": float(np.percentile(lengths, 95)),
                    f"{prefix}/max": float(np.max(lengths)),
                    f"{prefix}/truncation_rate": sum(
                        truncated for _, truncated in rows
                    )
                    / len(rows),
                }
            )

        metrics["mean_gen_tokens_per_sample"] = float(np.mean(all_lengths))

    if not missing_reward_samples and tagged_reward_samples:
        for environment in sorted(rewards_by_environment):
            values = rewards_by_environment[environment]
            metric_environment = _rollout_environment_metric_component(environment)
            prefix = f"{metric_environment}/reward"
            metrics[f"{prefix}/count"] = float(len(values))
            metrics.update(_scalar_summary(values, prefix))

    for environment in sorted(extras_by_environment):
        metric_environment = _rollout_environment_metric_component(environment)
        denominator = environment_sample_counts[environment]
        for metric_key in sorted(extras_by_environment[environment]):
            values = extras_by_environment[environment][metric_key]
            prefix = f"{metric_environment}/{metric_key}"
            metrics[f"{prefix}/count"] = float(len(values))
            metrics.update(
                _scalar_summary(
                    values,
                    prefix,
                    mean_denominator=denominator,
                )
            )
    return metrics


def _reduce_seq_logprob_error_metrics(
    records: list[dict[str, float]],
) -> dict[str, float]:
    """Reduce sequence-error metrics across streaming chunks."""

    def reduce_range(
        *,
        count_key: str,
        max_key: str,
        mean_key: str,
        min_key: str,
    ) -> dict[str, float]:
        populated = [record for record in records if record[count_key] > 0]
        count = sum(record[count_key] for record in populated)
        if not count:
            return {max_key: 0.0, mean_key: 0.0, min_key: 0.0}
        return {
            max_key: max(record[max_key] for record in populated),
            mean_key: sum(record[mean_key] * record[count_key] for record in populated)
            / count,
            min_key: min(record[min_key] for record in populated),
        }

    reduced = reduce_range(
        count_key="_num_valid_seqs_before",
        max_key="max_seq_mult_prob_error",
        mean_key="mean_seq_mult_prob_error",
        min_key="min_seq_mult_prob_error",
    )
    reduced.update(
        reduce_range(
            count_key="_num_valid_seqs_after",
            max_key="max_seq_mult_prob_error_after_mask",
            mean_key="mean_seq_mult_prob_error_after_mask",
            min_key="min_seq_mult_prob_error_after_mask",
        )
    )

    masked_count = sum(record["num_masked_seqs_by_logprob_error"] for record in records)
    reduced["num_masked_seqs_by_logprob_error"] = int(masked_count)
    reduced["masked_correct_pct"] = (
        sum(
            record["masked_correct_pct"] * record["num_masked_seqs_by_logprob_error"]
            for record in records
        )
        / masked_count
        if masked_count
        else 0.0
    )
    return reduced


def tensor_field(data: TensorDict, field_name: str) -> torch.Tensor:
    """Read a tensor column from a TensorDict, depadding if nested.

    Args:
        data: TensorDict returned by the data plane.
        field_name: Column name to fetch.

    Returns:
        Dense tensor (nested columns are padded with zeros).
    """
    value = data[field_name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected tensor field {field_name!r}; got {type(value)}")
    if value.is_nested:
        return torch.nested.to_padded_tensor(value, padding=0)
    return value


def squeeze_trailing_unit_dim(value: torch.Tensor) -> torch.Tensor:
    """Drop a trailing dim of size 1 if present.

    Args:
        value: Input tensor.

    Returns:
        Tensor without the trailing unit dim.
    """
    if value.dim() >= 2 and value.shape[-1] == 1:
        return value.squeeze(-1)
    return value


def fields_for_put(meta: KVBatchMeta, fields: dict[str, torch.Tensor]) -> TensorDict:
    """Pack tensors for DataPlane put, re-nesting jagged rows when needed.

    Args:
        meta: Batch meta whose sequence_lengths drive the nesting.
        fields: Field name to dense tensor.

    Returns:
        TensorDict shaped for dp_client.put_samples.
    """
    packed: dict[str, torch.Tensor] = {}
    if meta.sequence_lengths is None:
        for field_name, value in fields.items():
            packed[field_name] = value.detach().contiguous()
        # pyrefly: ignore[bad-argument-type]
        return TensorDict(packed, batch_size=[meta.size])

    lengths = torch.tensor(meta.sequence_lengths, dtype=torch.long)
    for field_name, value in fields.items():
        if value.dim() >= 2 and value.shape[1] == int(lengths.max().item()):
            rows = [
                value[i, : int(lengths[i].item())].detach().contiguous()
                for i in range(meta.size)
            ]
            packed[field_name] = torch.nested.as_nested_tensor(
                rows,
                layout=torch.jagged,
            )
        else:
            packed[field_name] = value.detach().contiguous()
    # pyrefly: ignore[bad-argument-type]
    return TensorDict(packed, batch_size=[meta.size])
