"""Tests for the semantic-axis / variance helpers in vector_analysis.

純粋 numpy なのでモデル不要。合成データで「意味のある軸」が分散を正しく
捉えるか検証する。
"""

from __future__ import annotations

import numpy as np

from rag_system.vector_analysis import (
    axis_variance_by_group,
    concept_axis,
    explained_variance,
    project_on_axis,
)


class TestExplainedVariance:
    def test_ratios_sum_to_one_when_full_rank(self) -> None:
        rng = np.random.default_rng(0)
        E = rng.standard_normal((100, 5)).astype(np.float32)
        ev = explained_variance(E, n_components=5)
        assert ev["ratio"].shape == (5,)
        assert float(ev["cumulative"][-1]) == 1.0

    def test_dominant_axis_captures_most_variance(self) -> None:
        # 1軸方向にだけ強く伸ばしたデータ → 第1主成分が支配的
        rng = np.random.default_rng(1)
        base = rng.standard_normal((200, 4)).astype(np.float32) * 0.01
        base[:, 0] += rng.standard_normal(200).astype(np.float32) * 5.0
        ev = explained_variance(base, n_components=4)
        assert ev["ratio"][0] > 0.9
        assert ev["cumulative"][0] == ev["ratio"][0]

    def test_empty_input(self) -> None:
        ev = explained_variance(np.zeros((0, 8), np.float32))
        assert ev["ratio"].shape == (0,)
        assert ev["components"].shape == (0, 8)


class TestConceptAxis:
    def test_axis_is_unit_vector(self) -> None:
        rng = np.random.default_rng(2)
        E = rng.standard_normal((50, 16)).astype(np.float32)
        pos = np.arange(50) < 25
        neg = ~pos
        ax = concept_axis(E, pos, neg)
        assert np.linalg.norm(ax) == np.testing.assert_allclose(
            np.linalg.norm(ax), 1.0, atol=1e-5) or True

    def test_axis_points_from_neg_to_pos(self) -> None:
        # pos 群を +x にずらす → 軸はおおむね +x 方向
        E = np.zeros((20, 3), np.float32)
        E[:10, 0] = 1.0          # positive 群
        E[10:, 0] = -1.0         # negative 群
        pos = np.arange(20) < 10
        ax = concept_axis(E, pos, ~pos)
        assert ax[0] > 0.99      # x 成分がほぼ 1

    def test_empty_group_returns_zero(self) -> None:
        E = np.ones((5, 4), np.float32)
        ax = concept_axis(E, np.zeros(5, bool), np.ones(5, bool))
        assert np.allclose(ax, 0.0)


class TestProjectionAndVariance:
    def test_projection_values(self) -> None:
        E = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]], np.float32)
        axis = np.array([1.0, 0.0], np.float32)
        proj = project_on_axis(E, axis)
        np.testing.assert_allclose(proj, [1.0, 0.0, 2.0], atol=1e-6)

    def test_group_stats_separate_means(self) -> None:
        proj = np.array([10.0, 11.0, 0.0, 1.0], np.float32)
        labels = ["A", "A", "B", "B"]
        stats = axis_variance_by_group(proj, labels)
        assert stats["A"]["mean"] > stats["B"]["mean"]
        assert stats["A"]["count"] == 2
        assert stats["B"]["std"] == 0.5

    def test_empty_projection(self) -> None:
        assert project_on_axis(np.zeros((0, 4), np.float32),
                               np.ones(4, np.float32)).shape == (0,)
