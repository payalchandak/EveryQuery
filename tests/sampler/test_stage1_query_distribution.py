"""Stage 1: ``QueryDistribution`` — the query draw (one code + one float-duration per query).

Pure function tests (determinism, bounds, unrounded floats, every validation error) plus
distribution-shape checks that bounds-only tests miss (log-uniform skews shorter; code draw covers
the universe).
"""

import numpy as np
import pytest
from omegaconf import OmegaConf

from every_query.generate_tasks.sample_tasks import QueryDistribution, QuerySpec
from every_query.utils.seeds import derive_seed


class TestQueryDistribution:
    """Unit tests for the redesigned Stage 1 query draw (``QueryDistribution``)."""

    def _rng(self, *parts):
        """Build an rng seeded on the ``queries`` axis, mirroring the redesign caller."""
        return np.random.default_rng(derive_seed(*parts, "queries"))

    def test_query_universe_size_is_derived(self, synthetic_query_codes):
        dist = QueryDistribution(synthetic_query_codes, 1.0, 365.0, "log-uniform")
        assert dist.query_universe_size == len(synthetic_query_codes)

    def test_from_config_builds_expected_dataclass(self, synthetic_query_codes):
        # query_codes is resolved inside from_config via read_query_codes(cfg.query_codes); an inline
        # list is returned order-preserving + deduped, so it matches synthetic_query_codes here.
        cfg = OmegaConf.create(
            {
                "query_codes": synthetic_query_codes,
                "min_duration": 1,
                "max_duration": 731,
                "duration_distribution": "log-uniform",
            }
        )
        dist = QueryDistribution.from_config(cfg)
        assert dist == QueryDistribution(synthetic_query_codes, 1.0, 731.0, "log-uniform")
        assert isinstance(dist.min_duration, float) and isinstance(dist.max_duration, float)

    def test_sample_determinism(self, synthetic_query_codes):
        dist = QueryDistribution(synthetic_query_codes, 1.0, 365.0, "log-uniform")
        a = dist.sample(32, self._rng(42))
        b = dist.sample(32, self._rng(42))
        assert a == b

    def test_sample_different_seeds_differ(self, synthetic_query_codes):
        dist = QueryDistribution(synthetic_query_codes, 1.0, 365.0, "log-uniform")
        a = dist.sample(32, self._rng(1))
        b = dist.sample(32, self._rng(2))
        assert a != b

    @pytest.mark.parametrize("distribution", ["uniform", "log-uniform"])
    def test_sample_respects_bounds(self, synthetic_query_codes, distribution):
        dist = QueryDistribution(synthetic_query_codes, 10.0, 100.0, distribution)
        specs = dist.sample(256, self._rng(7))
        assert all(isinstance(s, QuerySpec) for s in specs)
        assert all(s.code in synthetic_query_codes for s in specs)
        assert all(10.0 <= s.duration_days <= 100.0 for s in specs)

    @pytest.mark.parametrize("distribution", ["uniform", "log-uniform"])
    def test_durations_are_unrounded_floats(self, synthetic_query_codes, distribution):
        """Durations stay float — the intentional divergence from legacy whole-day quantization."""
        dist = QueryDistribution(synthetic_query_codes, 1.0, 731.0, distribution)
        specs = dist.sample(64, self._rng(3))
        assert all(isinstance(s.duration_days, float) for s in specs)
        assert any(s.duration_days != round(s.duration_days) for s in specs)

    def test_sample_zero_returns_empty(self, synthetic_query_codes):
        dist = QueryDistribution(synthetic_query_codes, 1.0, 365.0, "uniform")
        assert dist.sample(0, self._rng(0)) == []

    def test_sample_rejects_negative_num_queries(self, synthetic_query_codes):
        dist = QueryDistribution(synthetic_query_codes, 1.0, 365.0, "uniform")
        with pytest.raises(ValueError, match="num_queries"):
            dist.sample(-1, self._rng(0))

    def test_rejects_empty_query_codes(self):
        with pytest.raises(ValueError, match="query_codes"):
            QueryDistribution([], 1.0, 365.0, "uniform")

    def test_rejects_nonpositive_min_duration(self, synthetic_query_codes):
        with pytest.raises(ValueError, match="min_duration"):
            QueryDistribution(synthetic_query_codes, 0.0, 365.0, "uniform")

    def test_rejects_inverted_bounds(self, synthetic_query_codes):
        with pytest.raises(ValueError, match="max_duration"):
            QueryDistribution(synthetic_query_codes, 100.0, 50.0, "uniform")

    def test_rejects_unknown_distribution(self, synthetic_query_codes):
        with pytest.raises(ValueError, match="duration_distribution"):
            QueryDistribution(synthetic_query_codes, 1.0, 365.0, "normal")

    def test_sample_raises_if_distribution_drifts_from_valid_set(self, synthetic_query_codes):
        """``sample`` must not silently default to ``uniform`` if its branch drifts out of sync.

        with ``_VALID_DISTRIBUTIONS``. Simulates that drift via the frozen-dataclass escape hatch,
        since ``__post_init__`` already blocks constructing an invalid value normally.
        """
        dist = QueryDistribution(synthetic_query_codes, 1.0, 365.0, "uniform")
        object.__setattr__(dist, "duration_distribution", "bogus")
        with pytest.raises(AssertionError, match="duration_distribution"):
            dist.sample(1, self._rng(0))

    # -- Distribution-shape gap tests (plain pytest, fixed seed, generous bounds) --------------

    def test_log_uniform_skews_shorter_than_uniform(self, synthetic_query_codes):
        """Log-uniform preferentially samples shorter durations.

        Bounds tests only pin ``[min, max]`` containment; this pins the *shape* difference. Over a
        large sample on ``[1, 365]`` the uniform median sits near the arithmetic midpoint (183),
        while the log-uniform median sits near the geometric mean ``sqrt(1*365) ≈ 19`` — far below.
        """
        lo, hi, n = 1.0, 365.0, 5000
        uni = QueryDistribution(synthetic_query_codes, lo, hi, "uniform").sample(n, self._rng(123))
        logu = QueryDistribution(synthetic_query_codes, lo, hi, "log-uniform").sample(n, self._rng(123))
        uni_med = float(np.median([s.duration_days for s in uni]))
        logu_med = float(np.median([s.duration_days for s in logu]))
        midpoint = (lo + hi) / 2

        assert logu_med < uni_med, f"log-uniform median {logu_med} should be < uniform {uni_med}"
        assert logu_med < midpoint, f"log-uniform median {logu_med} should skew below {midpoint}"
        # Uniform median should center near the arithmetic midpoint (loose tolerance).
        assert abs(uni_med - midpoint) < 0.15 * (hi - lo)

    def test_code_draw_covers_universe(self, synthetic_query_codes):
        """Over a large draw, every code in the universe appears and none outside it ever does."""
        specs = QueryDistribution(synthetic_query_codes, 1.0, 365.0, "uniform").sample(2000, self._rng(5))
        drawn = {s.code for s in specs}
        assert drawn == set(synthetic_query_codes)
