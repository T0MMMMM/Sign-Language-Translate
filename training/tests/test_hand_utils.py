import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from hand_utils import extract_landmarks


class FakeHandLandmarks:
    def __init__(self, points):
        class LM:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z
        self.landmark = [LM(x, y, z) for x, y, z in points]


def make_hand(points):
    return FakeHandLandmarks(points)


def test_extract_landmarks_returns_63_values():
    pts = [(i * 0.05, i * 0.05, 0.0) for i in range(21)]
    hand = make_hand(pts)
    result = extract_landmarks(hand)
    assert result.shape == (63,)


def test_extract_landmarks_centered_on_wrist():
    pts = [(1.0, 1.0, 0.0)] + [(1.0 + 0.1 * i, 1.0, 0.0) for i in range(1, 21)]
    hand = make_hand(pts)
    result = extract_landmarks(hand)
    assert abs(result[0]) < 1e-6
    assert abs(result[1]) < 1e-6
    assert abs(result[2]) < 1e-6


def test_extract_landmarks_normalized():
    pts = [(0.0, 0.0, 0.0)] + [(0.1 * i, 0.0, 0.0) for i in range(1, 21)]
    hand = make_hand(pts)
    result = extract_landmarks(hand)
    assert np.all(np.abs(result) <= 1.0 + 1e-6)


def test_extract_landmarks_all_same_point():
    pts = [(0.5, 0.5, 0.0)] * 21
    hand = make_hand(pts)
    result = extract_landmarks(hand)
    assert np.allclose(result, 0.0)
