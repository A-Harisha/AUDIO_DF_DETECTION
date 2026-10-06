"""Segment IoU for evidence faithfulness."""
import pytest

from caesd.evidence import segment_iou


def test_identical_intervals():
    assert segment_iou([(0.0, 1.0)], [(0.0, 1.0)]) == pytest.approx(1.0)


def test_disjoint_intervals():
    assert segment_iou([(0.0, 1.0)], [(2.0, 3.0)]) == pytest.approx(0.0)


def test_half_overlap():
    assert segment_iou([(0.0, 2.0)], [(1.0, 3.0)]) == pytest.approx(1.0 / 3.0)
