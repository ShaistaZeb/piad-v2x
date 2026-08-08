"""Tests for the mock pseudonym manager (C5, W5)."""
import unittest

from piad_v2x.lifecycle.manager import MockPseudonymManager


class RotationRate(unittest.TestCase):
    def test_set_rotation_rate_records(self) -> None:
        m = MockPseudonymManager()
        m.set_rotation_rate(0.05, now=1.0)
        m.set_rotation_rate(0.1, now=2.0)
        self.assertEqual(m.current_rotation_rate, 0.1)
        self.assertEqual(m.n_rotation_events(), 2)
        self.assertEqual(m.rotation_history[0].time, 1.0)
        self.assertEqual(m.rotation_history[-1].rate, 0.1)


class SubmitHint(unittest.TestCase):
    def test_positive_hint_recorded(self) -> None:
        m = MockPseudonymManager()
        m.submit_hint("pk1", 0.7, now=10.0)
        self.assertEqual(m.n_hints(), 1)
        self.assertEqual(m.first_hint_time("pk1"), 10.0)

    def test_zero_hint_dropped(self) -> None:
        m = MockPseudonymManager()
        m.submit_hint("pk1", 0.0, now=5.0)
        m.submit_hint("pk1", -0.1, now=6.0)
        self.assertEqual(m.n_hints(), 0)
        self.assertIsNone(m.first_hint_time("pk1"))

    def test_first_hint_time_per_peer(self) -> None:
        m = MockPseudonymManager()
        m.submit_hint("pk1", 0.5, now=3.0)
        m.submit_hint("pk1", 0.6, now=4.0)
        m.submit_hint("pk2", 0.5, now=7.0)
        self.assertEqual(m.first_hint_time("pk1"), 3.0)
        self.assertEqual(m.first_hint_time("pk2"), 7.0)

    def test_hints_for_filters(self) -> None:
        m = MockPseudonymManager()
        m.submit_hint("pk1", 0.5, now=1.0)
        m.submit_hint("pk2", 0.5, now=2.0)
        self.assertEqual(len(m.hints_for("pk1")), 1)
        self.assertEqual(len(m.hints_for("pk2")), 1)
        self.assertEqual(len(m.hints_for("pk3")), 0)


if __name__ == "__main__":
    unittest.main()
