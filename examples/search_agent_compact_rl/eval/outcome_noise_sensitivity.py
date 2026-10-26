"""CPU-only sensitivity of Mosaic's group-relative trajectory advantages to flipped terminal outcomes.

Scope
-----
``runtime/reward.py`` falls back to an answer-F1 score when the LLM judge is
unavailable or unparseable, and
``baselines/session_aware_traj_level_dressage_broadcast.py`` turns the trajectory
reward into a binary outcome (``round`` -> 0/1) before forming success/failure
groups. One label change therefore moves a rollout between outcome subgroups and
re-centres/re-scales every trajectory advantage in the same prompt group. This
module measures that on a **fixed synthetic batch**, comparing clean, randomly
chosen, and outcome-correlated (asymmetric) label flips.

It drives the production entry point that the setting-2 Mosaic script registers,
``baselines/mosaic.py::convert_samples_to_train_data``, and additionally calls the
underlying ``_compute_session_aware_advantages`` so that per-record fields
(group type, outcome advantage, session signal) are available. The two paths are
asserted to agree, so every reported number is the implementation's own output.
No model, judge API, search tool, GPU stack, or training loop is used; ``torch``
on CPU is the only dependency. The session-level IG fusion stage is out of scope.

Count definitions (not interchangeable)
---------------------------------------
``changed_unflipped``    rollouts whose **own label was not flipped** whose
                         advantage moved by more than ``TOLERANCE`` (spillover).
``changed_flipped``      flipped rollouts whose own advantage moved.
``changed_total``        the two summed.
``unflipped_population`` all unflipped rollouts in the batch.
``unflipped_in_affected`` unflipped rollouts inside prompt groups that contain at
                         least one flip -- the denominator for the spillover share.

Statistical units
-----------------
Every advantage metric is computed at the **rollout (trajectory) level**: ``compare``
iterates the records that ``_compute_session_aware_advantages`` keys by
``rollout_id`` (``session_aware_traj_level_dressage_broadcast.py:213``), so
``changed_*``, ``*_population``, ``rollouts_in_affected_groups``, the ``delta_*``
magnitudes, ``zero_to_nonzero``, ``nonzero_to_zero``, ``sign_flips`` and
``max_abs_dev_reference`` all count rollouts -- never session samples. Rollouts are
still fanned out into ``num_sessions`` training samples, and two checks do run at
that sample level: the assertion that ``convert_samples_to_train_data`` writes every
sibling the advantage of its rollout record, and the implementation's own
``advantage/broadcast_consistency_error_count``, which counts rollout records whose
per-sample copies disagree. ``meta.num_session_samples`` is provenance only: no
exported metric uses it as a numerator or denominator.

Magnitudes are diluted differently depending on the population, so name it:
``delta_all_mean`` divides by the batch's rollouts, ``delta_unflipped_mean`` by
``unflipped_population`` (which includes rollouts in groups that contain no flip and
therefore contribute exactly 0), and ``delta_flipped_mean`` by ``flipped_population``.

Noise conditions
----------------
``clean``                  observed label == true label everywhere.
``random_symmetric``       ``n_flips`` rollouts picked uniformly over the batch,
                           irrespective of the true label.
``asymmetric_false_pos``   ``n_flips`` picked only among true failures.
``asymmetric_false_neg``   ``n_flips`` picked only among true successes.
``asymmetric_fp_skew3``    ``n_flips`` split 3:1 false-positive:false-negative
                           (rounded up for false positives).

Every non-clean condition injects the same flip count for a given
(layout, rate, seed), so random-vs-asymmetric is a pure direction-of-flip
contrast; ``flips_shortfall`` is nonzero only if a condition could not reach the
target because its label pool was too small.

The reference injection rate
----------------------------
``REFERENCE_INJECTION_RATE`` is the **average F1-fallback trigger rate the
maintainers reported** for their training run (Issue #1 comment 6407602910,
2026-10-26): the share of cases in which the LLM judge was replaced by the F1
fallback. It is *not* a measured label-error rate and *not* a general upper bound
on label noise. A fallback changes a binary outcome only when the F1 value rounds
to the opposite side of the judge's call, so the two event types differ; this
batch contains such label-preserving fallbacks and they are advantage-inert
(tested). Nothing here estimates how many labels actually changed in their run.

This sweep therefore reuses the reported number as a **reference injection rate**:
the same axis value, injected as flips. Rows are tagged in ``rate_kind``:

* ``reference_injection_rate`` - the injected flip count is
  ``round(0.000635 * batch size)``. Two distinct quantities must not be conflated:
  the *expected* number of reference events per batch (``0.000635 * batch size`` =
  0.081 at 128 rollouts, 4.88 at 7680) and the *injected* integer count produced
  by that rounding rule (0 at 128, 5 at 7680). The rounding rule first yields one
  injected flip at a batch of 788 rollouts, while an expected count of 1 needs
  1575. A 128-rollout step showing 0 injected flips is therefore an artefact of
  the counting rule at that batch size -- it is not evidence that a fallback
  cannot occur, or could not change a label, at that scale. The injected count
  corresponds to the reported incidence only under the convention of counting
  each fallback as one flipped label; that convention is a way of placing the
  reference point on the sweep axis, not a claim about their run.
* ``synthetic_stress_rate`` - chosen to probe sensitivity; no claim about how
  often such rates occur.
* ``no_noise_baseline`` - the clean batch.

Commands
--------
::

    # step geometry: 16 prompt groups x 8 rollouts = 128 rollouts, matching
    # --rollout-batch-size 16 --n-samples-per-prompt 8 of
    # scripts/train/run_setting_2_qwen3.5_4B_mosaic.sh
    python -m examples.search_agent_compact_rl.eval.outcome_noise_sensitivity --layout step --out ../out_step

    # run scale: 960 groups x 8 = 7680 rollouts, the full 60-step run of that
    # script -- the only layout whose injected flip count at the reference rate is a whole number
    python -m examples.search_agent_compact_rl.eval.outcome_noise_sensitivity --layout run --out ../out_run

    pytest tests/test_outcome_noise_sensitivity.py

Outputs (in ``--out``)
----------------------
``config.json`` (provenance, batch fingerprint, geometry, clean group-type
composition, ``rate_kind`` definitions, the once-per-layout parity negative
control, retained metric list) and ``per_seed.csv`` (one row per
condition x rate x seed). Both headline aggregates and per-seed values come from
that table; two consecutive runs are byte-identical.

Limitations
-----------
* Synthetic labels drawn from a per-prompt success-rate distribution, not real
  BrowseComp trajectories; no rollout dump is consumed.
* Flips are exact 1.0/0.0 reward changes, the sharpest form of a fallback-induced
  label change. The batch also holds partial-F1 rewards that round to the judge's
  own call; those are advantage-inert (tested), which is precisely why the reported
  fallback rate is not a label-flip rate.
* Advantages are a proxy for gradient direction. Nothing here measures training
  stability, sample efficiency, or final accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from examples.search_agent_compact_rl.baselines import mosaic
from examples.search_agent_compact_rl.baselines.session_aware_traj_level_dressage_broadcast import (
    _compute_session_aware_advantages,
)

from slime.utils.types import Sample

# The average F1-fallback trigger rate the maintainers reported for their training
# run in Issue #1 (comment 6407602910, 2026-10-26): the share of cases where the
# judge was replaced by F1. It is a fallback rate, not a measured label-error rate,
# and it is reused here only as a reference rate for injecting flips.
REFERENCE_INJECTION_RATE = 0.000635

# Defaults of scripts/train/run_setting_2_qwen3.5_4B_mosaic.sh
SESSION_AWARE_DEFAULT_ENV = {
    "SA_SESSION_AWARE_RHO": "0.2",
    "SA_SESSION_AWARE_BETA_0": "0.1",
    "SA_SESSION_AWARE_BETA_1": "0.2",
    "SA_SESSION_AWARE_SESSION_COUNT_KEY": "rollout_num_sessions",
}

LAYOUTS: dict[str, dict[str, Any]] = {
    "step": {
        "num_groups": 16,
        "rollouts_per_group": 8,
        "seeds": (0, 1, 2, 3, 4, 5, 6, 7),
        "rates": (0.0, REFERENCE_INJECTION_RATE, 0.01, 0.02, 0.05, 0.10, 0.20),
    },
    "run": {
        "num_groups": 960,
        "rollouts_per_group": 8,
        "seeds": (0, 1, 2, 3),
        "rates": (0.0, REFERENCE_INJECTION_RATE, 0.05, 0.20),
    },
}

CONDITIONS = (
    "clean",
    "random_symmetric",
    "asymmetric_false_pos",
    "asymmetric_false_neg",
    "asymmetric_fp_skew3",
)

NOISY_CONDITIONS = tuple(name for name in CONDITIONS if name != "clean")

GROUP_TYPES = ("all_failure", "all_success", "mixed")

RATE_KINDS = {
    "no_noise_baseline": "the clean batch, no label changed",
    "reference_injection_rate": (
        "the maintainers' reported average F1-fallback trigger rate reused as an injection rate: "
        "round(0.000635 * batch size) flips, i.e. the convention of counting each fallback as one flipped "
        "label. Not a measured label-error rate and not a general upper bound on label noise; a fallback "
        "changes a binary outcome only when the F1 value rounds to the opposite side. The expected event "
        "count and the injected integer count are different quantities, and neither is an estimate of label "
        "flips in the authors' run"
    ),
    "synthetic_stress_rate": "chosen to probe sensitivity; no claim about how often such rates occur",
}

TOLERANCE = 1e-5  # the implementation accumulates advantages in float32.

# Every column below is cited by a headline claim in the PR; the manifest records
# unlisted_numeric_columns so a metric cannot appear in the table unaccounted for.
RETAINED_METRICS = (
    "flips_injected",
    "flips_shortfall",
    "false_positives",
    "false_negatives",
    "groups_with_flips",
    "max_flips_in_one_group",
    "unflipped_partial_fallback",
    "num_prompt_groups",
    "groups_clean_all_failure",
    "groups_clean_all_success",
    "groups_clean_mixed",
    "transition_from_all_failure",
    "transition_from_all_success",
    "transition_from_mixed",
    "transition_rate_from_all_failure",
    "transition_rate_from_all_success",
    "transition_rate_from_mixed",
    "groups_affected",
    "groups_changed_type",
    "rollouts_in_affected_groups",
    "unflipped_in_affected",
    "changed_unflipped",
    "changed_flipped",
    "changed_total",
    "unflipped_population",
    "flipped_population",
    "sign_flips",
    "sign_flips_unflipped",
    "zero_to_nonzero",
    "nonzero_to_zero",
    "delta_all_mean",
    "delta_all_median",
    "delta_all_max",
    "delta_unflipped_mean",
    "delta_unflipped_max",
    "delta_flipped_mean",
    "locality_violations",
    "impl_sign_flip_count",
    "impl_broadcast_consistency_error_count",
    "n_flips_planned",
    "max_abs_dev_reference",
)


def rate_kind(rate: float) -> str:
    if rate == 0.0:
        return "no_noise_baseline"
    if rate == REFERENCE_INJECTION_RATE:
        return "reference_injection_rate"
    return "synthetic_stress_rate"


def reference_rate_geometry(num_rollouts: int) -> dict[str, Any]:
    """Keep the three different readings of the reference rate apart.

    * expected events: the rate times the batch size, a real number;
    * injected flips: what this sweep actually injects, ``round(rate * batch)``;
    * thresholds: the batch size at which the rounding rule first injects one flip
      versus the batch size at which the expected count reaches one.

    A batch where the injected count rounds to 0 says nothing about whether a
    fallback, or a label change, can occur at that size.
    """
    smallest_with_one_injection = 1
    while int(round(REFERENCE_INJECTION_RATE * smallest_with_one_injection)) < 1:
        smallest_with_one_injection += 1
    return {
        "rate": REFERENCE_INJECTION_RATE,
        "rate_is_a_reported_average_fallback_trigger_rate": True,
        "rate_is_a_measured_label_error_rate": False,
        "rate_is_a_general_upper_bound_on_label_noise": False,
        "num_rollouts": num_rollouts,
        "expected_reference_events": REFERENCE_INJECTION_RATE * num_rollouts,
        "injected_flips_under_rounding_rule": int(round(REFERENCE_INJECTION_RATE * num_rollouts)),
        "smallest_batch_with_one_injected_flip": smallest_with_one_injection,
        "batch_where_expected_events_reach_one": int(1 / REFERENCE_INJECTION_RATE) + 1,
        "zero_injections_is_only_a_rounding_artefact": True,
    }


# ---------------------------------------------------------------------------
# Fixed synthetic batch
# ---------------------------------------------------------------------------


def build_batch(*, num_groups: int, rollouts_per_group: int, batch_seed: int) -> list[dict[str, Any]]:
    """Draw the fixed base rollouts: true outcome, observed reward, prompt group, session counts.

    ``p_success`` 0.0 and 1.0 are deliberate: they produce all-failure /
    all-success prompt groups, where Mosaic switches to its ``beta_0`` / ``beta_1``
    session-only branches. ``fallback_like`` rewards stand in for an F1 fallback
    that agrees with the judge after rounding; they are the batch's negative
    controls.
    """
    rng = random.Random(f"batch|{batch_seed}|{num_groups}x{rollouts_per_group}")
    p_success_grid = (0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0)
    p_weights = (0.10, 0.15, 0.15, 0.20, 0.15, 0.15, 0.10)
    session_counts = (1, 2, 3, 4)
    session_weights = (0.45, 0.30, 0.15, 0.10)

    rollouts: list[dict[str, Any]] = []
    for group_index in range(num_groups):
        p_success = rng.choices(p_success_grid, weights=p_weights, k=1)[0]
        for _ in range(rollouts_per_group):
            true_label = 1 if rng.random() < p_success else 0
            if rng.random() < 0.15:
                low, high = (0.56, 0.94) if true_label == 1 else (0.06, 0.44)
                observed_reward = round(rng.uniform(low, high), 3)
                fallback_like = True
            else:
                observed_reward = float(true_label)
                fallback_like = False
            num_sessions = rng.choices(session_counts, weights=session_weights, k=1)[0]
            rollouts.append(
                {
                    "rollout_id": len(rollouts),
                    "group_index": group_index,
                    "true_label": true_label,
                    "observed_reward": observed_reward,
                    "binary_reward": float(round(observed_reward)),
                    "num_sessions": num_sessions,
                    "fallback_like": fallback_like,
                }
            )
    return rollouts


def batch_fingerprint(rollouts: list[dict[str, Any]]) -> str:
    payload = json.dumps(rollouts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Label flips
# ---------------------------------------------------------------------------


def flip_indices(*, condition: str, n_flips: int, labels: list[int], rng: random.Random) -> tuple[list[int], str]:
    """Pick the rollouts whose binary outcome is inverted; return (indices, note)."""
    if condition == "clean" or n_flips == 0:
        return [], ""
    failures = [i for i, label in enumerate(labels) if label == 0]
    successes = [i for i, label in enumerate(labels) if label == 1]

    if condition == "random_symmetric":
        return sorted(rng.sample(range(len(labels)), n_flips)), ""
    if condition == "asymmetric_false_pos":
        pool = failures
    elif condition == "asymmetric_false_neg":
        pool = successes
    elif condition == "asymmetric_fp_skew3":
        # ceil() rather than round(): round() is banker's rounding in Python, which
        # would silently make the 3:1 split depend on the parity of n_flips.
        fp = min(-(-3 * n_flips // 4), len(failures))
        fn = min(n_flips - fp, len(successes))
        fp = min(n_flips - fn, len(failures))
        picked = rng.sample(failures, fp) + rng.sample(successes, fn)
        return sorted(picked), "" if fp + fn == n_flips else "pool_limited"
    else:
        raise ValueError(f"unknown condition {condition!r}")
    if len(pool) < n_flips:
        return sorted(pool), "pool_limited"
    return sorted(rng.sample(pool, n_flips)), ""


def apply_noise(
    rollouts: list[dict[str, Any]], condition: str, n_flips: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return noisy copies of the rollouts plus the injection bookkeeping."""
    labels = [rollout["true_label"] for rollout in rollouts]
    rng = random.Random(f"flip|{condition}|{n_flips}|{seed}")
    picked, note = flip_indices(condition=condition, n_flips=n_flips, labels=labels, rng=rng)
    picked_set = set(picked)

    noisy: list[dict[str, Any]] = []
    per_group_flips: dict[int, int] = {}
    false_pos = false_neg = 0
    for index, rollout in enumerate(rollouts):
        flipped = index in picked_set
        if flipped:
            observed_reward = float(1 - rollout["true_label"])
            if rollout["true_label"] == 0:
                false_pos += 1
            else:
                false_neg += 1
            per_group_flips[rollout["group_index"]] = per_group_flips.get(rollout["group_index"], 0) + 1
        else:
            observed_reward = rollout["observed_reward"]
        item = dict(rollout)
        item["observed_reward"] = observed_reward
        item["binary_reward"] = float(round(observed_reward))
        item["label_flipped"] = flipped
        noisy.append(item)

    effective = sum(1 for item in noisy if item["binary_reward"] != float(item["true_label"]))
    if effective != len(picked):
        raise AssertionError(f"injected {len(picked)} flips but {effective} changed a binary label")
    bookkeeping = {
        "flips_injected": len(picked),
        "flips_shortfall": n_flips - len(picked),
        "false_positives": false_pos,
        "false_negatives": false_neg,
        "groups_with_flips": len(per_group_flips),
        "max_flips_in_one_group": max(per_group_flips.values()) if per_group_flips else 0,
        "pool_limited": note,
        "unflipped_partial_fallback": sum(1 for item in noisy if not item["label_flipped"] and item["fallback_like"]),
    }
    return noisy, bookkeeping


# ---------------------------------------------------------------------------
# Driving the production implementation
# ---------------------------------------------------------------------------


def make_args(num_items: int) -> SimpleNamespace:
    # Mirrors the setting-2 Mosaic run: --advantage-estimator grpo,
    # --n-samples-per-prompt 8 (so std normalisation stays on), no --reward-key.
    return SimpleNamespace(
        reward_key=None,
        advantage_estimator="grpo",
        grpo_std_normalization=True,
        global_batch_size=num_items,
        custom_reward_post_process_path=None,
    )


def make_samples(rollouts: list[dict[str, Any]], *, response_length: int = 4) -> list[Sample]:
    """Fan each rollout out into its session samples, as the agent loop does.

    Siblings of one rollout share ``rollout_id`` and carry the same trajectory
    reward, which is what makes the implementation's broadcast-consistency check
    non-vacuous: it compares the advantage written to every sibling.
    """
    samples: list[Sample] = []
    for rollout in rollouts:
        for _ in range(rollout["num_sessions"]):
            samples.append(
                Sample(
                    group_index=rollout["group_index"],
                    index=len(samples),
                    rollout_id=rollout["rollout_id"],
                    tokens=[0] * (response_length + 1),
                    response_length=response_length,
                    reward=rollout["observed_reward"],
                    loss_mask=[1] * response_length,
                    status=Sample.Status.COMPLETED,
                    metadata={"trajectory_raw_reward": rollout["observed_reward"]},
                    train_metadata={"rollout_num_sessions": rollout["num_sessions"]},
                )
            )
    return samples


def run_production(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    """Run the production advantage code on ``rollouts`` and cross-check both entry points."""
    args = make_args(len(rollouts))
    samples = make_samples(rollouts)
    _, _, metrics, records = _compute_session_aware_advantages(args, samples)
    train_data = mosaic.convert_samples_to_train_data(args, samples)

    if len(train_data["rewards"]) != len(samples):
        raise AssertionError("broadcast length mismatch")
    if metrics["advantage/broadcast_consistency_error_count"] != 0.0:
        raise AssertionError("implementation reported a broadcast consistency error")
    for position, sample in enumerate(samples):
        expected = records[sample.rollout_id].traj_advantage
        if abs(train_data["rewards"][position] - expected) > TOLERANCE:
            raise AssertionError(
                f"convert_samples_to_train_data disagrees with records for rollout {sample.rollout_id}"
            )

    return {
        "records": records,
        "metrics": metrics,
        "advantage": {rollout_id: record.traj_advantage for rollout_id, record in records.items()},
    }


# ---------------------------------------------------------------------------
# Independent reference (parity gate only; reported numbers come from the real code)
# ---------------------------------------------------------------------------


def reference_trajectory_advantages(
    rollouts: list[dict[str, Any]], *, rho: float, beta_0: float, beta_1: float, mutate: str | None = None
) -> dict[int, float]:
    """Re-derive the session-aware trajectory advantage with pure-Python statistics.

    Written from the mechanism description rather than by calling the
    implementation, and used only to check that this analysis reads the code the
    same way twice. ``mutate`` introduces a deliberate deviation for the negative
    control: ``"unbiased_std"`` uses the sample instead of the population standard
    deviation, and ``"uncentered_signal"`` drops the ``[-1, 1]`` centring of the
    session-count rank signal.
    """
    groups: dict[int, list[dict[str, Any]]] = {}
    for rollout in rollouts:
        groups.setdefault(rollout["group_index"], []).append(rollout)

    def rank_signal(values: list[float]) -> list[float]:
        if len(values) <= 1 or len(set(values)) <= 1:
            return [0.0] * len(values)
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position + 1
            while end < len(order) and values[order[end]] == values[order[position]]:
                end += 1
            average_rank = (position + 1 + end) / 2.0
            for k in order[position:end]:
                ranks[k] = average_rank
            position = end
        denom = len(values) - 1
        if mutate == "uncentered_signal":
            return [(rank - 1.0) / denom for rank in ranks]
        return [2.0 * ((rank - 1.0) / denom) - 1.0 for rank in ranks]

    advantages: dict[int, float] = {}
    for group in groups.values():
        binary = [float(round(rollout["observed_reward"])) for rollout in group]
        successes = sum(binary)
        mixed = 0.0 < successes < len(group)
        outcome = [0.0] * len(group)
        if mixed:
            mean = statistics.fmean(binary)
            if mutate == "unbiased_std":
                std = statistics.stdev(binary) if len(binary) > 1 else 0.0
            else:
                std = statistics.pstdev(binary)
            outcome = [(value - mean) / (std + 1e-6) for value in binary]
        signals = [0.0] * len(group)
        for reward_value in (0.0, 1.0):
            members = [i for i, value in enumerate(binary) if value == reward_value]
            if members:
                values = [float(group[i]["num_sessions"]) for i in members]
                for i, signal in zip(members, rank_signal(values), strict=True):
                    signals[i] = signal
        for i, rollout in enumerate(group):
            if mixed:
                value = (1.0 - rho * signals[i]) * outcome[i]
            elif successes == len(group):
                value = -beta_1 * signals[i]
            else:
                value = beta_0 * min(signals[i], 0.0)
            advantages[rollout["rollout_id"]] = value
    return advantages


def max_deviation_vs_reference(
    rollouts: list[dict[str, Any]], advantage: dict[int, float], config: dict[str, float]
) -> float:
    """Largest absolute gap between the implementation and the reference on this batch."""
    reference = reference_trajectory_advantages(rollouts, **config)
    return max(abs(reference[rollout_id] - value) for rollout_id, value in advantage.items())


def negative_control(
    rollouts: list[dict[str, Any]], advantage: dict[int, float], config: dict[str, float]
) -> dict[str, float]:
    """The same reference with one formula detail changed; it must NOT be in parity.

    Computed once per layout on the clean batch: the deviations depend on the
    batch, not on which labels were flipped.
    """
    baseline = max_deviation_vs_reference(rollouts, advantage, config)
    out = {"max_abs_dev_reference_clean_batch": baseline}
    for mutate in ("unbiased_std", "uncentered_signal"):
        mutated = reference_trajectory_advantages(rollouts, **config, mutate=mutate)
        out[f"max_abs_dev_mutate_{mutate}"] = max(
            abs(mutated[rollout_id] - value) for rollout_id, value in advantage.items()
        )
    threshold = 100 * max(baseline, TOLERANCE)
    out["mutated_min_abs_dev"] = min(
        out["max_abs_dev_mutate_unbiased_std"], out["max_abs_dev_mutate_uncentered_signal"]
    )
    out["gate_is_live"] = float(out["mutated_min_abs_dev"] > threshold)
    return out


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _sign(value: float) -> int:
    if value > TOLERANCE:
        return 1
    if value < -TOLERANCE:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------


def compare(clean: dict[str, Any], noisy: dict[str, Any]) -> dict[str, Any]:
    """Diff two runs of the production code on the same batch, split by flipped vs unflipped."""
    clean_records: dict[Any, Any] = clean["records"]
    noisy_records: dict[Any, Any] = noisy["records"]

    affected_groups: set[int] = set()
    for rollout_id, record in noisy_records.items():
        if record.binary_reward != clean_records[rollout_id].binary_reward:
            affected_groups.add(record.group_index)

    delta_all: list[float] = []
    delta_unflipped: list[float] = []
    delta_flipped: list[float] = []
    changed_unflipped = changed_flipped = flipped_population = unflipped_population = 0
    sign_flips = sign_flips_unflipped = zero_to_nonzero = nonzero_to_zero = 0
    in_affected = 0
    locality_violations = 0

    for rollout_id, record in noisy_records.items():
        base = clean_records[rollout_id]
        delta = abs(record.traj_advantage - base.traj_advantage)
        delta_all.append(delta)
        flipped = record.binary_reward != base.binary_reward
        (delta_flipped if flipped else delta_unflipped).append(delta)
        if flipped:
            flipped_population += 1
            changed_flipped += int(delta > TOLERANCE)
        else:
            unflipped_population += 1
            changed_unflipped += int(delta > TOLERANCE)
        if record.group_index in affected_groups:
            in_affected += 1
        elif delta > TOLERANCE:
            locality_violations += 1

        base_sign, noisy_sign = _sign(base.traj_advantage), _sign(record.traj_advantage)
        if base_sign != noisy_sign:
            if base_sign and noisy_sign:
                sign_flips += 1
                sign_flips_unflipped += int(not flipped)
            elif noisy_sign == 0:
                nonzero_to_zero += 1
            else:
                zero_to_nonzero += 1

    clean_group_type: dict[int, str] = {}
    noisy_group_type: dict[int, str] = {}
    for record in clean_records.values():
        if clean_group_type.setdefault(record.group_index, record.group_type) != record.group_type:
            raise AssertionError(f"prompt group {record.group_index} has inconsistent clean types")
    for record in noisy_records.values():
        if noisy_group_type.setdefault(record.group_index, record.group_type) != record.group_type:
            raise AssertionError(f"prompt group {record.group_index} has inconsistent noisy types")

    out: dict[str, Any] = {
        "num_prompt_groups": float(len(clean_group_type)),
        "groups_affected": float(len(affected_groups)),
        "rollouts_in_affected_groups": float(in_affected),
        "unflipped_in_affected": float(in_affected - flipped_population),
        "changed_unflipped": float(changed_unflipped),
        "changed_flipped": float(changed_flipped),
        "changed_total": float(changed_unflipped + changed_flipped),
        "unflipped_population": float(unflipped_population),
        "flipped_population": float(flipped_population),
        "sign_flips": float(sign_flips),
        "sign_flips_unflipped": float(sign_flips_unflipped),
        "zero_to_nonzero": float(zero_to_nonzero),
        "nonzero_to_zero": float(nonzero_to_zero),
        "delta_all_mean": _mean(delta_all),
        "delta_all_median": _median(delta_all),
        "delta_all_max": max(delta_all, default=0.0),
        "delta_unflipped_mean": _mean(delta_unflipped),
        "delta_unflipped_max": max(delta_unflipped, default=0.0),
        "delta_flipped_mean": _mean(delta_flipped),
        "groups_changed_type": float(
            sum(1 for group, base in clean_group_type.items() if base != noisy_group_type[group])
        ),
        "locality_violations": float(locality_violations),
        "impl_sign_flip_count": float(noisy["metrics"].get("advantage/sign_flip_count", 0.0)),
        "impl_broadcast_consistency_error_count": float(
            noisy["metrics"].get("advantage/broadcast_consistency_error_count", 0.0)
        ),
    }
    # Prompt-group type changes, as rates over the number of groups that started in that
    # type: a false positive can only move an all-failure group and a false negative only
    # an all-success one, so raw counts are not comparable otherwise.
    for group_type in GROUP_TYPES:
        denominator = float(sum(1 for value in clean_group_type.values() if value == group_type))
        changed = float(
            sum(
                1 for group, base in clean_group_type.items() if base == group_type and base != noisy_group_type[group]
            )
        )
        out[f"groups_clean_{group_type}"] = denominator
        out[f"transition_from_{group_type}"] = changed
        out[f"transition_rate_from_{group_type}"] = changed / denominator if denominator else 0.0
    return out


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run_one(
    *,
    layout: str,
    rollouts: list[dict[str, Any]],
    clean: dict[str, Any],
    condition: str,
    rate: float,
    seed: int,
    config: dict[str, float],
) -> dict[str, Any]:
    n_flips = int(round(rate * len(rollouts)))
    noisy_rollouts, bookkeeping = apply_noise(rollouts, condition, n_flips, seed)
    noisy = run_production(noisy_rollouts)
    comparison = compare(clean, noisy)
    if comparison["locality_violations"]:
        raise AssertionError(
            f"{condition} rate={rate} seed={seed}: {comparison['locality_violations']:.0f} rollouts in "
            "unflipped prompt groups changed advantage"
        )
    return {
        "layout": layout,
        "condition": condition,
        "rate": rate,
        "rate_kind": rate_kind(rate),
        "seed": seed,
        "n_flips_planned": n_flips,
        **bookkeeping,
        **comparison,
        "max_abs_dev_reference": max_deviation_vs_reference(noisy_rollouts, noisy["advantage"], config),
    }


def sweep(
    *,
    layout: str,
    batch_seed: int,
    rates: tuple[float, ...] | None,
    seeds: tuple[int, ...] | None,
    conditions: tuple[str, ...],
    config: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = LAYOUTS[layout]
    rollouts = build_batch(
        num_groups=spec["num_groups"], rollouts_per_group=spec["rollouts_per_group"], batch_seed=batch_seed
    )
    rate_table = rates or spec["rates"]
    seed_table = seeds or spec["seeds"]

    clean = run_production(rollouts)
    by_group: dict[int, str] = {}
    for record in clean["records"].values():
        if by_group.setdefault(record.group_index, record.group_type) != record.group_type:
            raise AssertionError("inconsistent clean group type within a prompt group")
    clean_prompt_types = {name: sum(1 for value in by_group.values() if value == name) for name in GROUP_TYPES}

    meta = {
        "layout": layout,
        "num_groups": spec["num_groups"],
        "rollouts_per_group": spec["rollouts_per_group"],
        "num_rollouts": len(rollouts),
        "num_session_samples": sum(rollout["num_sessions"] for rollout in rollouts),
        "batch_fingerprint": batch_fingerprint(rollouts),
        "batch_seed": batch_seed,
        "seeds": list(seed_table),
        "rates": list(rate_table),
        "clean_prompt_group_type_counts": clean_prompt_types,
        "clean_true_label_count": sum(rollout["true_label"] for rollout in rollouts),
        "clean_fallback_like_count": sum(1 for rollout in rollouts if rollout["fallback_like"]),
        "reference_rate_geometry": reference_rate_geometry(len(rollouts)),
        "negative_control": negative_control(rollouts, clean["advantage"], config),
    }

    runs: list[dict[str, Any]] = []
    for condition in conditions:
        for rate in rate_table:
            # "clean" ignores the rate, so repeating it would only duplicate the zero-flip baseline.
            if condition == "clean" and rate != 0.0:
                continue
            for seed in seed_table:
                runs.append(
                    run_one(
                        layout=layout,
                        rollouts=rollouts,
                        clean=clean,
                        condition=condition,
                        rate=rate,
                        seed=seed,
                        config=config,
                    )
                )
    return runs, meta


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

IDENTITY_COLUMNS = ("layout", "condition", "rate", "rate_kind", "seed")

# pool_limited is a string flag whose numeric effect is already recorded as flips_shortfall.
FLATTEN_SKIP = ("pool_limited",)


def flatten(record: dict[str, Any], *, prefix: str = "", skip: tuple[str, ...] = ()) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in record.items():
        path = f"{prefix}{key}"
        if path in skip or isinstance(value, (bool, str)) or value is None:
            continue
        if isinstance(value, (int, float)):
            out[path] = float(value)
        elif isinstance(value, dict):
            out.update(flatten(value, prefix=f"{path}.", skip=skip))
    return out


def write_seed_csv(out_path: Path, runs: list[dict[str, Any]]) -> list[str]:
    """Write one row per (condition, rate, seed); columns are derived from the records."""
    rows = [{key: record[key] for key in IDENTITY_COLUMNS} | flatten(record, skip=FLATTEN_SKIP) for record in runs]
    columns = list(IDENTITY_COLUMNS) + sorted({key for row in rows for key in row} - set(IDENTITY_COLUMNS))
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return columns


SUMMARY_FIELDS = (
    "flips_injected",
    "false_positives",
    "false_negatives",
    "groups_with_flips",
    "groups_affected",
    "unflipped_in_affected",
    "changed_unflipped",
    "changed_flipped",
    "changed_total",
    "sign_flips",
    "sign_flips_unflipped",
    "zero_to_nonzero",
    "delta_all_mean",
    "delta_all_max",
    "transition_rate_from_all_failure",
    "transition_rate_from_all_success",
    "max_abs_dev_reference",
)

SUMMARY_HEADERS = (
    "flips",
    "FP",
    "FN",
    "grpFlip",
    "grpAff",
    "unflAff",
    "chgUnfl",
    "chgFlip",
    "chgTot",
    "sign",
    "signUnf",
    "z2nz",
    "dAmean",
    "dAmax",
    "rtFail",
    "rtSucc",
    "parity",
)

SUMMARY_WIDTHS = (5, 4, 4, 7, 6, 7, 8, 7, 7, 5, 7, 5, 7, 6, 6, 6, 9)


def summarise(runs: list[dict[str, Any]]) -> dict[tuple[str, float], dict[str, float]]:
    """Mean over seeds, per (condition, rate), for the printed headline table."""
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for record in runs:
        grouped.setdefault((record["condition"], record["rate"]), []).append(record)
    return {
        key: {name: _mean([row[name] for row in group]) for name in SUMMARY_FIELDS} | {"seeds": float(len(group))}
        for key, group in grouped.items()
    }


def print_summary(runs: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    print(
        f"layout={meta['layout']} rollouts={meta['num_rollouts']} samples={meta['num_session_samples']} "
        f"records={len(runs)} fingerprint={meta['batch_fingerprint'][:12]} "
        f"clean_prompt_types={meta['clean_prompt_group_type_counts']} "
        f"true_success={meta['clean_true_label_count']}/{meta['num_rollouts']} "
        f"fallback_like={meta['clean_fallback_like_count']}"
    )
    control = meta["negative_control"]
    print(
        f"parity: clean batch |reference - implementation| = {control['max_abs_dev_reference_clean_batch']:.3e}; "
        f"mutated references deviate by {control['max_abs_dev_mutate_unbiased_std']:.3e} (unbiased std) and "
        f"{control['max_abs_dev_mutate_uncentered_signal']:.3e} (uncentred signal); "
        f"gate_is_live={bool(control['gate_is_live'])}"
    )
    print(
        f"locality_violations total = {sum(r['locality_violations'] for r in runs):.0f} ; "
        f"impl_sign_flip_count max = {max(r['impl_sign_flip_count'] for r in runs):.0f} -- the multiplicative "
        "session scale reverses no direction by construction (1 - rho*signal > 0 for rho < 1), so the sign flips "
        "below come from the group-relative comparison changing; this analysis does not separate re-centring from "
        "the changed session-count ranking, and it does not test the session-level gating"
    )
    geometry = meta["reference_rate_geometry"]
    reference = [r for r in runs if r["rate_kind"] == "reference_injection_rate"]
    if reference:
        injected = geometry["injected_flips_under_rounding_rule"]
        if injected == 0:
            framing = (
                f"expected {geometry['expected_reference_events']:.3f} events per batch, but the "
                "round(rate*N) injection rule gives 0 injected flips, so every advantage equals the clean batch"
            )
        else:
            framing = (
                f"expected {geometry['expected_reference_events']:.3f} events per batch, "
                f"{injected:.0f} injected flips -> "
                f"{min(r['changed_unflipped'] for r in reference):.0f}-"
                f"{max(r['changed_unflipped'] for r in reference):.0f} changed unflipped siblings (of "
                f"{min(r['unflipped_in_affected'] for r in reference):.0f} unflipped rollouts in the affected "
                f"groups) + {injected:.0f} flipped = {min(r['changed_total'] for r in reference):.0f}-"
                f"{max(r['changed_total'] for r in reference):.0f} changed in total"
            )
        print(
            f"reference injection rate ({REFERENCE_INJECTION_RATE:.5%}, the reported average F1-fallback trigger "
            f"rate): {framing}. The rate is not a measured label-error rate and not a general upper bound on "
            f"label noise; a fallback changes a binary outcome only when the F1 rounds to the opposite side. "
            f"Rounding thresholds: one injected flip from {geometry['smallest_batch_with_one_injected_flip']} "
            f"rollouts, one expected event from {geometry['batch_where_expected_events_reach_one']}; neither says "
            "a fallback cannot occur at smaller batch sizes."
        )
    print(
        f"{'condition':22s} {'rate':>9s} "
        + " ".join(f"{header:>{width}s}" for header, width in zip(SUMMARY_HEADERS, SUMMARY_WIDTHS, strict=True))
    )
    for (condition, rate), values in sorted(summarise(runs).items()):
        cells = " ".join(
            (
                f"{values[field]:>{width}.1e}"
                if field == "max_abs_dev_reference"
                else f"{values[field]:>{width}.0f}" if field == "flips_injected" else f"{values[field]:>{width}.3f}"
            )
            for field, width in zip(SUMMARY_FIELDS, SUMMARY_WIDTHS, strict=True)
        )
        print(f"{condition:22s} {rate:9.5f} {cells}")


def _git(*cmd: str) -> str:
    try:
        return subprocess.run(["git", *cmd], capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance only, never fatal
        return "unknown"


def write_outputs(
    out_dir: Path,
    runs: list[dict[str, Any]],
    meta: dict[str, Any],
    config: dict[str, float],
    args: argparse.Namespace,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = write_seed_csv(out_dir / "per_seed.csv", runs)
    unreported = sorted(set(columns) - set(IDENTITY_COLUMNS) - set(RETAINED_METRICS))
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join([sys.executable, "-m", __name__, *sys.argv[1:]]),
        "purpose": "synthetic CPU-only sensitivity of group-relative trajectory advantages to flipped outcomes",
        "issue": "https://github.com/LARK-AI-Lab/Mosaic/issues/1",
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "worktree_dirty": bool(_git("status", "--porcelain")),
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "session_aware_config": config,
        "session_aware_config_source": "run_setting_2_qwen3.5_4B_mosaic.sh defaults",
        "advantage_source": (
            "examples.search_agent_compact_rl.baselines.mosaic.convert_samples_to_train_data, the entry point "
            "run_setting_2_qwen3.5_4B_mosaic.sh registers, cross-checked against _compute_session_aware_advantages; "
            "the pure-Python reference is only a parity gate"
        ),
        "reference_injection_rate": REFERENCE_INJECTION_RATE,
        "reference_rate_provenance": (
            "the average F1-fallback trigger rate the maintainers reported in Issue #1 comment 6407602910 "
            "(2026-10-26): the share of cases where the judge was replaced by F1"
        ),
        "reference_rate_is_a_measured_label_error_rate": False,
        "reference_rate_is_a_general_upper_bound_on_label_noise": False,
        "fallback_does_not_necessarily_change_the_binary_outcome": True,
        "expected_events_and_injected_counts_are_distinct": True,
        "rate_kind_definitions": RATE_KINDS,
        "rates_note": (
            "the 0.0635% row reuses the reported fallback rate as an injection rate: injected flips = "
            "round(rate * batch size), the convention of counting each fallback as one flipped label. It is "
            "neither a measured label-error rate nor a general upper bound on label noise, and the injected "
            "count is not an estimate of labels changed in the authors' run. See meta.reference_rate_geometry "
            "for the expected count, the injected count, and the two different thresholds"
        ),
        "retained_metrics": list(RETAINED_METRICS),
        "unlisted_numeric_columns": unreported,
        "csv_columns": columns,
        "meta": meta,
        "args": {key: value for key, value in vars(args).items() if key != "out"},
    }
    (out_dir / "config.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return columns


def read_config() -> dict[str, float]:
    return {
        "rho": float(os.environ["SA_SESSION_AWARE_RHO"]),
        "beta_0": float(os.environ["SA_SESSION_AWARE_BETA_0"]),
        "beta_1": float(os.environ["SA_SESSION_AWARE_BETA_1"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CPU-only outcome-label flip sensitivity for Mosaic group-relative advantages"
    )
    parser.add_argument("--layout", choices=tuple(LAYOUTS), default="step")
    parser.add_argument("--out", required=True, help="output directory (created; keep it outside the repo)")
    parser.add_argument("--batch-seed", type=int, default=20261016)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--rates", default="", help="comma-separated override of the layout rate table")
    parser.add_argument("--seeds", default="", help="comma-separated override of the layout seed table")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    for name, value in SESSION_AWARE_DEFAULT_ENV.items():
        os.environ.setdefault(name, value)

    runs, meta = sweep(
        layout=args.layout,
        batch_seed=args.batch_seed,
        rates=tuple(float(item) for item in args.rates.split(",")) if args.rates else None,
        seeds=tuple(int(item) for item in args.seeds.split(",")) if args.seeds else None,
        conditions=tuple(item.strip() for item in args.conditions.split(",") if item.strip()),
        config=read_config(),
    )
    write_outputs(Path(args.out), runs, meta, read_config(), args)
    if not args.quiet:
        print_summary(runs, meta)
        print(f"wrote {args.out}/config.json per_seed.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
