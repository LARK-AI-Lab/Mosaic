"""Tests for the CPU-only outcome-label flip sensitivity analysis.

The parity tests compare the production Mosaic advantage code against an
independent pure-Python re-derivation, and the negative control shows that the
parity check can actually fail. No model, API, or GPU stack is used.
"""

import math
from pathlib import Path

import pytest

from examples.search_agent_compact_rl.baselines import mosaic
from examples.search_agent_compact_rl.eval import outcome_noise_sensitivity as ons

CONFIG = {"rho": 0.2, "beta_0": 0.1, "beta_1": 0.2}

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def session_aware_environment(monkeypatch):
    for name, value in ons.SESSION_AWARE_DEFAULT_ENV.items():
        monkeypatch.setenv(name, value)


def _batch(num_groups=16, rollouts_per_group=8, batch_seed=7):
    return ons.build_batch(num_groups=num_groups, rollouts_per_group=rollouts_per_group, batch_seed=batch_seed)


def _rollouts(spec):
    """spec: list of (group_index, reward, num_sessions) with distinct rollout ids."""
    return [
        {
            "rollout_id": index,
            "group_index": group,
            "true_label": int(round(reward)),
            "observed_reward": reward,
            "binary_reward": float(round(reward)),
            "num_sessions": sessions,
            "fallback_like": not float(reward).is_integer(),
        }
        for index, (group, reward, sessions) in enumerate(spec)
    ]


def _sweep(**kwargs):
    defaults = {
        "layout": "step",
        "batch_seed": 5,
        "rates": (0.0, 0.05),
        "seeds": (0, 1),
        "conditions": ons.CONDITIONS,
        "config": CONFIG,
    }
    defaults.update(kwargs)
    return ons.sweep(**defaults)


def test_analysis_drives_the_entry_point_the_training_script_registers():
    script = (
        REPO_ROOT / "examples/search_agent_compact_rl/scripts/train/run_setting_2_qwen3.5_4B_mosaic.sh"
    ).read_text(encoding="utf-8")
    assert "--custom-convert-samples-to-train-data-path examples.search_agent_compact_rl.baselines.mosaic" in script
    assert ons.mosaic.convert_samples_to_train_data is mosaic.convert_samples_to_train_data


def test_reference_matches_the_production_implementation():
    rollouts = _batch()
    advantage = ons.run_production(rollouts)["advantage"]
    assert ons.max_deviation_vs_reference(rollouts, advantage, CONFIG) < 1e-5


@pytest.mark.parametrize("mutate", ["unbiased_std", "uncentered_signal"])
def test_parity_gate_is_not_vacuous(mutate):
    """A reference with one formula detail changed must disagree, so the parity test can fail."""
    rollouts = _batch()
    advantage = ons.run_production(rollouts)["advantage"]
    mutated = ons.reference_trajectory_advantages(rollouts, **CONFIG, mutate=mutate)
    deviation = max(abs(mutated[rollout_id] - value) for rollout_id, value in advantage.items())
    assert deviation > 1e-3


def test_negative_control_reports_a_live_gate():
    _, meta = _sweep(rates=(0.0,), seeds=(0,), conditions=("clean",))
    control = meta["negative_control"]
    assert control["gate_is_live"] == 1.0
    assert control["max_abs_dev_reference_clean_batch"] < 1e-5
    assert control["mutated_min_abs_dev"] > 100 * control["max_abs_dev_reference_clean_batch"]


def test_batch_is_fixed_across_conditions_and_seeds():
    rollouts = _batch()
    first = ons.batch_fingerprint(rollouts)
    assert first == ons.batch_fingerprint(_batch())
    assert first != ons.batch_fingerprint(_batch(batch_seed=1))
    assert first != ons.batch_fingerprint(_batch(num_groups=8))
    for condition in ons.NOISY_CONDITIONS:
        noisy, bookkeeping = ons.apply_noise(rollouts, condition, 6, seed=3)
        assert len(noisy) == len(rollouts)
        assert bookkeeping["flips_injected"] == 6
        assert bookkeeping["false_positives"] + bookkeeping["false_negatives"] == 6
        per_group = {}
        for rollout in noisy:
            if rollout["label_flipped"]:
                per_group[rollout["group_index"]] = per_group.get(rollout["group_index"], 0) + 1
        assert sum(per_group.values()) == 6
        assert bookkeeping["groups_with_flips"] == len(per_group)
        assert bookkeeping["max_flips_in_one_group"] == max(per_group.values())
        changed = [
            i
            for i, (a, b) in enumerate(zip(rollouts, noisy, strict=True))
            if a["observed_reward"] != b["observed_reward"]
        ]
        assert len(changed) == 6
        assert all(noisy[i]["binary_reward"] != float(noisy[i]["true_label"]) for i in changed)
        assert all(a["group_index"] == b["group_index"] for a, b in zip(rollouts, noisy, strict=True))
        assert all(a["num_sessions"] == b["num_sessions"] for a, b in zip(rollouts, noisy, strict=True))


def test_total_flip_rate_is_matched_across_noisy_conditions():
    """Random and outcome-correlated conditions differ only in the direction of the flips."""
    rollouts = _batch(num_groups=16, rollouts_per_group=8)
    for rate in (0.01, 0.02, 0.05, 0.10, 0.20):
        counts = set()
        for condition in ons.NOISY_CONDITIONS:
            _, bookkeeping = ons.apply_noise(rollouts, condition, int(round(rate * len(rollouts))), seed=11)
            assert bookkeeping["pool_limited"] == "", f"{condition} at {rate} could not reach the target count"
            counts.add(bookkeeping["flips_injected"] / len(rollouts))
        assert len(counts) == 1, f"conditions disagree on total flip rate at {rate}: {counts}"


def test_asymmetric_conditions_are_outcome_correlated():
    rollouts = _batch()
    expectations = {
        "asymmetric_false_pos": (6, 0),
        "asymmetric_false_neg": (0, 6),
        "asymmetric_fp_skew3": (5, 1),
    }
    for condition, (fp, fn) in expectations.items():
        _, bookkeeping = ons.apply_noise(rollouts, condition, 6, seed=0)
        assert (bookkeeping["false_positives"], bookkeeping["false_negatives"]) == (fp, fn)
    _, bookkeeping = ons.apply_noise(rollouts, "random_symmetric", 6, seed=0)
    assert 0 <= bookkeeping["false_positives"] <= 6
    assert bookkeeping["false_positives"] + bookkeeping["false_negatives"] == 6


def test_reference_rate_expected_events_injected_count_and_thresholds_are_distinct():
    """round(rate * N) is a counting rule, not a statement about what can occur at that size."""
    step = ons.LAYOUTS["step"]
    assert step["num_groups"] * step["rollouts_per_group"] == 128
    geometry = ons.reference_rate_geometry(128)
    assert geometry["rate"] == ons.REFERENCE_INJECTION_RATE
    assert geometry["expected_reference_events"] == pytest.approx(0.08128, abs=1e-6)
    assert geometry["injected_flips_under_rounding_rule"] == 0
    # The rounding rule injects the first flip well before the expected count reaches one.
    assert geometry["smallest_batch_with_one_injected_flip"] == 788
    assert geometry["batch_where_expected_events_reach_one"] == 1575
    assert geometry["smallest_batch_with_one_injected_flip"] < geometry["batch_where_expected_events_reach_one"]
    assert geometry["rate_is_a_measured_label_error_rate"] is False
    assert geometry["rate_is_a_general_upper_bound_on_label_noise"] is False

    run_geometry = ons.reference_rate_geometry(
        ons.LAYOUTS["run"]["num_groups"] * ons.LAYOUTS["run"]["rollouts_per_group"]
    )
    assert run_geometry["expected_reference_events"] == pytest.approx(4.8768, abs=1e-4)
    assert run_geometry["injected_flips_under_rounding_rule"] == 5


def test_rate_kind_labels_the_reference_injection_rate():
    assert ons.rate_kind(0.0) == "no_noise_baseline"
    assert ons.rate_kind(ons.REFERENCE_INJECTION_RATE) == "reference_injection_rate"
    assert ons.rate_kind(0.05) == "synthetic_stress_rate"
    definition = ons.RATE_KINDS["reference_injection_rate"]
    assert "a measured label-error rate" in definition.lower()
    assert "not a general upper bound on label noise" in definition
    assert "round(0.000635 * batch size)" in definition
    runs, _ = _sweep(
        rates=(0.0, ons.REFERENCE_INJECTION_RATE, 0.05),
        seeds=(0,),
        conditions=("clean", "random_symmetric"),
    )
    assert {record["rate_kind"] for record in runs} == set(ons.RATE_KINDS)
    for record in runs:
        if record["rate_kind"] == "reference_injection_rate":
            assert record["flips_injected"] == 0
            assert record["changed_total"] == 0.0


def test_changed_counts_decompose_into_flipped_and_unflipped():
    runs, meta = _sweep(rates=(0.05, 0.20), seeds=(0, 1), conditions=ons.NOISY_CONDITIONS)
    for record in runs:
        flips = record["flips_injected"]
        assert record["changed_total"] == record["changed_flipped"] + record["changed_unflipped"]
        assert record["changed_flipped"] <= flips
        assert record["flipped_population"] == flips
        assert record["unflipped_population"] == meta["num_rollouts"] - flips
        assert record["unflipped_in_affected"] == record["rollouts_in_affected_groups"] - flips
        assert record["changed_unflipped"] <= record["unflipped_in_affected"]
        assert record["sign_flips_unflipped"] <= record["changed_unflipped"]


def test_clean_record_is_an_exact_copy_of_the_baseline():
    runs, _ = _sweep(rates=(0.0,), seeds=(0,), conditions=("clean",))
    record = runs[0]
    assert record["changed_total"] == 0.0
    assert record["groups_changed_type"] == 0.0
    assert record["sign_flips"] == 0.0


def test_noise_stays_inside_its_prompt_group():
    """Only the flipped prompt group may change; every other group must be identical."""
    rollouts = _batch(num_groups=8, rollouts_per_group=4)
    clean = ons.run_production(rollouts)
    noisy_rollouts = [dict(rollout) for rollout in rollouts]
    index = next(i for i, rollout in enumerate(noisy_rollouts) if i % 4 == 0 and i != 0)
    noisy_rollouts[index]["observed_reward"] = float(1 - rollouts[index]["true_label"])
    noisy_rollouts[index]["binary_reward"] = float(round(noisy_rollouts[index]["observed_reward"]))
    noisy = ons.run_production(noisy_rollouts)

    for rollout_id, record in noisy["records"].items():
        base = clean["records"][rollout_id]
        if base.group_index == rollouts[index]["group_index"]:
            continue
        assert math.isclose(base.traj_advantage, record.traj_advantage, rel_tol=1e-9, abs_tol=1e-9)
    changed = [
        rollout_id
        for rollout_id, record in noisy["records"].items()
        if clean["records"][rollout_id].group_index == rollouts[index]["group_index"]
        and not math.isclose(record.traj_advantage, clean["records"][rollout_id].traj_advantage, abs_tol=1e-6)
    ]
    assert changed, "the flipped group should have changed"


def test_single_flip_rewrites_untouched_siblings_and_can_flip_their_sign():
    rollouts = _rollouts([(0, 1.0, 1), (0, 1.0, 2), (0, 1.0, 3), (0, 1.0, 4)])
    clean = ons.run_production(rollouts)["advantage"]
    # An all-success group carries no outcome contrast, so credit is purely the session-count signal.
    assert math.isclose(clean[0], 0.2, abs_tol=1e-5)
    assert math.isclose(clean[3], -0.2, abs_tol=1e-5)

    noisy_rollouts = [dict(rollout) for rollout in rollouts]
    noisy_rollouts[0]["observed_reward"] = 0.0
    noisy_rollouts[0]["binary_reward"] = 0.0
    noisy = ons.run_production(noisy_rollouts)["advantage"]

    assert noisy[0] < 0.0  # the flipped rollout leaves the all-success branch for the mixed one
    untouched = [i for i in noisy if i != 0]
    assert all(abs(noisy[i] - clean[i]) > 1e-2 for i in untouched)
    assert any(noisy[i] * clean[i] < 0.0 for i in untouched), "expected an untouched sibling to flip sign"


def test_partial_f1_that_rounds_to_the_same_label_is_advantage_inert():
    """A fallback event is not a label flip, which is why a fallback incidence is an upper bound."""
    integer = _rollouts([(0, 1.0, 2), (0, 0.0, 3), (1, 1.0, 1), (1, 0.0, 4)])
    partial = _rollouts([(0, 0.62, 2), (0, 0.31, 3), (1, 0.87, 1), (1, 0.44, 4)])
    clean = ons.run_production(integer)["advantage"]
    noisy = ons.run_production(partial)["advantage"]
    for rollout_id in clean:
        assert math.isclose(clean[rollout_id], noisy[rollout_id], rel_tol=1e-6, abs_tol=1e-6)


def test_sweep_is_deterministic_across_calls():
    first, first_meta = _sweep(rates=(0.05,), seeds=(0, 1), conditions=("random_symmetric", "asymmetric_false_pos"))
    second, second_meta = _sweep(rates=(0.05,), seeds=(0, 1), conditions=("random_symmetric", "asymmetric_false_pos"))
    assert first == second
    assert first_meta == second_meta


def test_csv_columns_are_derived_and_all_declared(tmp_path):
    import csv as csv_module

    runs, _ = _sweep(rates=(0.0, 0.2), seeds=(0,), conditions=("clean", "asymmetric_false_pos"))
    columns = ons.write_seed_csv(tmp_path / "per_seed.csv", runs)
    assert list(columns[: len(ons.IDENTITY_COLUMNS)]) == list(ons.IDENTITY_COLUMNS)
    assert sorted(set(columns) - set(ons.IDENTITY_COLUMNS)) == sorted(ons.RETAINED_METRICS)
    assert "per_group_flips" not in columns and "pool_limited" not in columns
    with (tmp_path / "per_seed.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv_module.DictReader(handle))
    assert len(rows) == len(runs)
    assert all(row["condition"] and row["rate_kind"] for row in rows)


def test_run_production_rejects_a_broadcast_mismatch():
    """The two-path cross-check must be able to fail, otherwise it proves nothing."""
    rollouts = _batch(num_groups=2, rollouts_per_group=4)
    samples = ons.make_samples(rollouts)
    assert len(samples) > len(rollouts), "rollouts must fan out so the broadcast check is not vacuous"
    first = samples[0].rollout_id
    siblings = [sample for sample in samples if sample.rollout_id == first]
    assert len(siblings) == rollouts[first]["num_sessions"]

    real = mosaic.convert_samples_to_train_data
    try:

        def broken(args, samples_arg):
            data = real(args, samples_arg)
            data["rewards"] = list(data["rewards"])
            data["rewards"][siblings[-1].index] += 1.0
            return data

        mosaic.convert_samples_to_train_data = broken
        with pytest.raises(AssertionError, match="disagrees with records"):
            ons.run_production(rollouts)
    finally:
        mosaic.convert_samples_to_train_data = real

    assert ons.run_production(rollouts)["advantage"]  # still works once restored
