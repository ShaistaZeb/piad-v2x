"""Tests for the adversary generators + the simulator's phi_t hook."""
from __future__ import annotations

import unittest

from piad_v2x.attacks.adversaries import (
    cold_start_gaming,
    coordinated_rotation_dos,
    honest_traffic,
    merge_streams,
    persistent_attacker,
    staged_retirement_sybil,
    staggered_rotation_dos,
    stealthy_attacker,
)
from piad_v2x.lifecycle.simulator import OracleTrustProvider, stream_messages


class Generators(unittest.TestCase):
    def test_honest_traffic_classes_all_zero(self) -> None:
        rows = honest_traffic(n_peers=3, duration_s=1.0)
        self.assertGreater(len(rows), 0)
        self.assertTrue(all(r["class"] == 0 for r in rows))

    def test_persistent_attacker_all_attack(self) -> None:
        rows = persistent_attacker(pk=9001, duration_s=2.0)
        self.assertTrue(all(r["class"] != 0 for r in rows))
        self.assertTrue(all(r["senderPseudo"] == 9001 for r in rows))

    def test_stealthy_attacker_mixes_classes(self) -> None:
        rows = stealthy_attacker(pk=9002, duration_s=2.0,
                                 burst_attack=3, burst_benign=7)
        classes = {r["class"] for r in rows}
        self.assertIn(0, classes)
        self.assertGreater(len(classes), 1)

    def test_rotation_dos_uses_n_distinct_pseudonyms(self) -> None:
        rows = coordinated_rotation_dos(n_sybils=12, duration_s=1.0)
        self.assertEqual(len({r["senderPseudo"] for r in rows}), 12)

    def test_cold_start_gaming_below_w_half(self) -> None:
        """Each pseudonym should appear < 25 times (default w/2 gate)."""
        rows = cold_start_gaming(n_rotations=5, msgs_per_pseudonym=24)
        counts: dict[int, int] = {}
        for r in rows:
            counts[r["senderPseudo"]] = counts.get(r["senderPseudo"], 0) + 1
        self.assertTrue(all(c < 25 for c in counts.values()))
        self.assertEqual(len(counts), 5)

    def test_merge_streams_sorted_by_send_time(self) -> None:
        a = persistent_attacker(pk=1, duration_s=1.0, start_t=0.5)
        b = persistent_attacker(pk=2, duration_s=1.0, start_t=0.0)
        merged = merge_streams(a, b)
        times = [r["sendTime"] for r in merged]
        self.assertEqual(times, sorted(times))


class PhysicsFlagHook(unittest.TestCase):
    def test_physics_flag_records_for_every_message(self) -> None:
        rows = honest_traffic(n_peers=3, duration_s=1.0)
        report, _ = stream_messages(rows, OracleTrustProvider())
        self.assertEqual(len(report.physics_flag_events), len(rows))

    def test_phi_t_fires_under_sudden_mass_injection(self) -> None:
        """A burst of fresh pseudonyms appearing rapidly should produce a
        positive ∂ρ/∂t, exceeding the default 5/s threshold."""
        # 1 honest peer for 5 seconds (low density baseline)
        background = honest_traffic(n_peers=1, duration_s=5.0)
        # 30 Sybils appearing in the next 1 second
        burst = coordinated_rotation_dos(
            n_sybils=30, duration_s=1.0, start_t=5.0,
        )
        rows = merge_streams(background, burst)
        report, _ = stream_messages(rows, OracleTrustProvider())
        self.assertGreater(report.n_physics_flags, 0,
                           "phi_t did not fire under coordinated mass injection")

    def test_phi_t_does_not_fire_in_steady_state(self) -> None:
        """Steady honest traffic should not trip the phi_t flag."""
        rows = honest_traffic(n_peers=5, duration_s=20.0)
        report, _ = stream_messages(rows, OracleTrustProvider())
        # In honest steady state ρ_t is constant; flag should not fire.
        self.assertEqual(report.n_physics_flags, 0)

    def test_cadence_trace_optional(self) -> None:
        rows = honest_traffic(n_peers=2, duration_s=0.5)
        report_off, _ = stream_messages(rows, OracleTrustProvider(),
                                        record_cadence_trace=False)
        report_on, _ = stream_messages(rows, OracleTrustProvider(),
                                       record_cadence_trace=True)
        self.assertEqual(len(report_off.cadence_trace), 0)
        self.assertEqual(len(report_on.cadence_trace), len(rows))


class StaggeredAndStagedAdversaries(unittest.TestCase):
    """Staggered and staged-retirement Sybil adversaries."""

    def test_staggered_rotation_dos_arrival_count(self) -> None:
        rows = staggered_rotation_dos(
            n_sybils=20, inter_arrival_s=1.0, duration_s=25.0
        )
        pks = {r["senderPseudo"] for r in rows}
        # 20 sybils in 25s @ 1.0s ia -> all 20 should appear
        self.assertEqual(len(pks), 20)

    def test_staggered_rotation_dos_truncates_at_duration(self) -> None:
        rows = staggered_rotation_dos(
            n_sybils=20, inter_arrival_s=2.0, duration_s=10.0
        )
        pks = {r["senderPseudo"] for r in rows}
        # 20 * 2s ia = 40s required; only 5 should fit in 10s
        self.assertEqual(len(pks), 5)

    def test_staged_retirement_n_concurrent_active_at_once(self) -> None:
        rows = staged_retirement_sybil(
            n_concurrent=3, rotation_period_s=5.0, duration_s=15.0,
        )
        # At t=2.5s (mid-first-cohort), exactly 3 distinct senders should
        # have broadcast.
        early = {r["senderPseudo"] for r in rows
                 if 2.0 <= r["sendTime"] <= 2.5}
        self.assertEqual(len(early), 3)

    def test_staged_retirement_total_unique_grows_with_rotations(self) -> None:
        rows = staged_retirement_sybil(
            n_concurrent=3, rotation_period_s=5.0, duration_s=25.0,
        )
        pks = {r["senderPseudo"] for r in rows}
        # n_concurrent=3, 5 cohorts in 25s (at t=0,5,10,15,20) -> 15 unique
        self.assertEqual(len(pks), 15)

    def test_staged_retirement_all_attack_class(self) -> None:
        rows = staged_retirement_sybil(
            n_concurrent=2, rotation_period_s=5.0, duration_s=10.0,
        )
        self.assertTrue(all(r["class"] != 0 for r in rows))


if __name__ == "__main__":
    unittest.main()
