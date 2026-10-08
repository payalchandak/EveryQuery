"""Prevalence-weighted boundary/start pools and prefix exclusions in the multitask sampler.

Drawn uniformly over a whole cohort vocabulary, almost every event boundary is a code that never
recurs, so almost every event-bounded window runs to the end of the timeline and almost every
event-defined start leaves an empty window.  ``code_weighting: prevalence`` is the knob that fixes
that; these tests pin the parts of it a downstream label bit depends on.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from every_query.generate_tasks.sample_multitask_sequences import (
    BoundaryDistribution,
    _apply_prefix_exclusions,
    build_code_weights,
    build_target_vocabulary,
    config_fingerprint,
    effective_support,
    read_boundary_codes,
    read_exclude_prefixes,
    resolve_boundary_pools,
)
from tests.multitask.conftest import CODES, K, base_cfg, make_codes_parquet

WEIGHTED = ["A", "B", "C", "D"]


def _weighted_cohort(tmp_path: Path, occurrences: list[int], codes: list[str] = WEIGHTED) -> Path:
    """A metadata root whose ``codes.parquet`` carries the prevalence column the weighting reads."""
    root = tmp_path / "cohort"
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "code": codes,
            "code/vocab_index": list(range(1, len(codes) + 1)),
            "code/n_occurrences": occurrences,
            "code/n_subjects": [max(1, o // 2) for o in occurrences],
        }
    ).write_parquet(root / "metadata" / "codes.parquet")
    return root


# --- weights --------------------------------------------------------------------------------------


def test_weights_are_proportional_and_normalized(tmp_path: Path):
    root = _weighted_cohort(tmp_path, [100, 200, 300, 400])
    w = build_code_weights(root, WEIGHTED, "code/n_occurrences", 1.0)
    assert pytest.approx(sum(w)) == 1.0
    assert w == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_power_tempers_the_distribution(tmp_path: Path):
    root = _weighted_cohort(tmp_path, [1, 100, 10_000, 1_000_000])
    flat = effective_support(build_code_weights(root, WEIGHTED, "code/n_occurrences", 0.0), 4)
    damped = effective_support(build_code_weights(root, WEIGHTED, "code/n_occurrences", 0.5), 4)
    sharp = effective_support(build_code_weights(root, WEIGHTED, "code/n_occurrences", 1.0), 4)
    assert flat == pytest.approx(4.0)
    assert 1.0 < sharp < damped < flat


def test_zero_and_null_statistics_are_floored_not_dropped(tmp_path: Path):
    """A code the cohort never saw must stay drawable, or the pool silently shrinks."""
    root = tmp_path / "cohort"
    (root / "metadata").mkdir(parents=True)
    pl.DataFrame(
        {
            "code": WEIGHTED,
            "code/vocab_index": [1, 2, 3, 4],
            "code/n_occurrences": [0, None, 5, 95],
        }
    ).write_parquet(root / "metadata" / "codes.parquet")
    w = build_code_weights(root, WEIGHTED, "code/n_occurrences", 1.0)
    assert all(x > 0 for x in w)
    assert w[0] == w[1] == pytest.approx(w[2])  # both floored to the smallest positive weight


def test_unknown_code_raises(tmp_path: Path):
    root = _weighted_cohort(tmp_path, [1, 1, 1, 1])
    with pytest.raises(ValueError, match="absent from"):
        build_code_weights(root, [*WEIGHTED, "MISSING"], "code/n_occurrences", 1.0)


def test_negative_power_raises(tmp_path: Path):
    root = _weighted_cohort(tmp_path, [1, 1, 1, 1])
    with pytest.raises(ValueError, match="code_weight_power"):
        build_code_weights(root, WEIGHTED, "code/n_occurrences", -1.0)


# --- prefix exclusions ----------------------------------------------------------------------------


def test_exclusions_apply_to_the_default_pool(tmp_path: Path):
    make_codes_parquet(tmp_path, [*CODES, "TIMELINE//DELTA//1h", "TIMELINE//DELTA//1d"])
    vocab = build_target_vocabulary(tmp_path)
    full = read_boundary_codes(None, vocab)
    trimmed = read_boundary_codes(None, vocab, ("TIMELINE//DELTA",))
    assert len(full) - len(trimmed) == 2
    assert not any(c.startswith("TIMELINE//DELTA") for c in trimmed)
    # TIMELINE//END shares no prefix with the delta tokens and must survive.
    assert "TIMELINE//END" in trimmed


def test_exclusions_apply_to_an_explicit_pool(tmp_path: Path):
    make_codes_parquet(tmp_path, [*CODES, "TIMELINE//DELTA//1h"])
    vocab = build_target_vocabulary(tmp_path)
    pool = read_boundary_codes(["C//1", "TIMELINE//DELTA//1h"], vocab, ("TIMELINE//DELTA",))
    assert pool == ["C//1"]


def test_excluding_everything_raises(tmp_path: Path):
    make_codes_parquet(tmp_path, ["C//1", "C//2"])
    vocab = build_target_vocabulary(tmp_path)
    with pytest.raises(ValueError, match="empties the boundary pool"):
        read_boundary_codes(None, vocab, ("C//",))


@pytest.mark.parametrize(
    ("spec", "expected"),
    [(None, ()), ("A//", ("A//",)), (["A//", "B//"], ("A//", "B//"))],
)
def test_read_exclude_prefixes(spec, expected):
    assert read_exclude_prefixes(spec) == expected


def test_apply_prefix_exclusions_is_order_preserving():
    assert _apply_prefix_exclusions(["b", "a", "x//1"], ("x//",), "boundary") == ["b", "a"]


# --- the draw ---------------------------------------------------------------------------------


def _dist(weights: tuple[float, ...]) -> BoundaryDistribution:
    return BoundaryDistribution(
        num_bounds=2,
        min_duration=1.0,
        max_duration=10.0,
        duration_distribution="uniform",
        eventbound_fraction=1.0,
        boundary_codes=tuple(WEIGHTED),
        condition_codes=tuple(WEIGHTED),
        boundary_weights=weights,
    )


def _rngs():
    return [np.random.default_rng(i) for i in range(7)]


def test_degenerate_weights_draw_one_code():
    sample = _dist((0.0, 1.0, 0.0, 0.0)).sample(50, *_rngs())
    assert set(sample.bound_events.ravel().tolist()) == {"B"}


def test_weighted_draw_tracks_the_weights():
    sample = _dist((0.7, 0.1, 0.1, 0.1)).sample(4000, *_rngs())
    share = (sample.bound_events == "A").mean()
    assert 0.65 < share < 0.75


def test_uniform_is_the_default():
    sample = _dist(()).sample(4000, *_rngs())
    shares = [float((sample.bound_events == c).mean()) for c in WEIGHTED]
    assert all(0.2 < s < 0.3 for s in shares)


def test_start_events_are_weighted_independently():
    dist = BoundaryDistribution(
        num_bounds=2,
        min_duration=1.0,
        max_duration=10.0,
        duration_distribution="uniform",
        eventbound_fraction=1.0,
        boundary_codes=tuple(WEIGHTED),
        condition_codes=tuple(WEIGHTED),
        eventstart_fraction=1.0,
        prediction_time_start_fraction=0.0,
        start_event_codes=tuple(WEIGHTED),
        boundary_weights=(1.0, 0.0, 0.0, 0.0),
        start_event_weights=(0.0, 0.0, 0.0, 1.0),
    )
    sample = dist.sample(30, *_rngs())
    assert set(sample.bound_events.ravel().tolist()) == {"A"}
    assert set(sample.start_events.ravel().tolist()) == {"D"}


@pytest.mark.parametrize("bad", [(0.5,), (0.5, 0.5, 0.5, 0.5, 0.5), (1.0, -1.0, 0.5, 0.5)])
def test_malformed_weights_raise(bad):
    with pytest.raises(ValueError, match="weights"):
        _dist(tuple(bad))


def test_unnormalized_weights_raise():
    with pytest.raises(ValueError, match="normalized"):
        _dist((0.25, 0.25, 0.25, 0.5))


# --- config plumbing + provenance ---------------------------------------------------------------


def test_resolve_boundary_pools_off_by_default(tmp_path: Path):
    root = _weighted_cohort(tmp_path, [1, 2, 3, 4])
    vocab = build_target_vocabulary(root)
    cfg = OmegaConf.create({"query_codes": str(root)})
    codes, w, start_codes, sw = resolve_boundary_pools(cfg, vocab)
    assert codes == start_codes == WEIGHTED
    assert w == () and sw == ()


def test_resolve_boundary_pools_weighted(tmp_path: Path):
    root = _weighted_cohort(tmp_path, [100, 200, 300, 400])
    vocab = build_target_vocabulary(root)
    cfg = OmegaConf.create(
        {"query_codes": str(root), "code_weighting": "prevalence", "code_weight_power": 1.0}
    )
    _, w, _, sw = resolve_boundary_pools(cfg, vocab)
    assert w == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert sw == pytest.approx(w)


def test_resolve_boundary_pools_rejects_unknown_policy(tmp_path: Path):
    root = _weighted_cohort(tmp_path, [1, 1, 1, 1])
    vocab = build_target_vocabulary(root)
    cfg = OmegaConf.create({"query_codes": str(root), "code_weighting": "inverse"})
    with pytest.raises(ValueError, match="code_weighting"):
        resolve_boundary_pools(cfg, vocab)


def test_weights_change_the_config_fingerprint(tmp_path: Path):
    """Same pool, different weights => different labels, so the fingerprint must differ or a rerun into an
    existing out_dir would silently reuse the other arm's shards."""
    make_codes_parquet(tmp_path, WEIGHTED)
    vocab = build_target_vocabulary(tmp_path)
    uniform = _dist(())
    weighted = _dist((0.7, 0.1, 0.1, 0.1))
    assert config_fingerprint(uniform, vocab) != config_fingerprint(weighted, vocab)


def test_end_to_end_weighted_run_records_its_policy(tmp_path: Path, synthetic_cohort: Path):
    """A weighted run must label, and its manifest must say the pool was weighted."""
    from every_query.generate_tasks import sample_multitask_sequences as sms

    # The synthetic cohort's codes.parquet has no prevalence column; add one with a single dominant
    # code so the drawn boundaries are predictable.
    meta_fp = synthetic_cohort / "metadata" / "codes.parquet"
    meta = pl.read_parquet(meta_fp)
    occ = [1] * meta.height
    occ[0] = 10_000
    meta.with_columns(pl.Series("code/n_occurrences", occ)).write_parquet(meta_fp)

    out_dir = tmp_path / "labels"
    cfg = OmegaConf.create(
        base_cfg(
            synthetic_cohort,
            out_dir,
            num_training_examples=40,
            code_weighting="prevalence",
            code_weight_column="code/n_occurrences",
            code_weight_power=1.0,
        )
    )
    sms.run(cfg)

    manifest = sms.read_manifest(out_dir / "train")
    assert manifest["boundary_code_policy"] == "weighted"
    assert manifest["start_event_code_policy"] == "weighted"
    assert manifest["num_bounds"] == K

    bounds = pl.concat([pl.read_parquet(fp) for fp in sorted((out_dir / "train").glob("*.parquet"))])[
        "bound_events"
    ].explode()
    drawn = set(bounds.drop_nulls().unique().to_list())
    assert drawn, "no event-bounded slot was drawn"
    # The dominant code carries ~99.9% of the mass, so every drawn boundary should be it.
    assert drawn == {meta["code"][0]}
