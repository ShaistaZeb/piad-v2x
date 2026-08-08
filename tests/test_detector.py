"""Tests for the W3 multi-physics PINN detector."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from piad_v2x.models.detector import (
    DetectorConfig,
    PINNDetector,
    kinematic_violation,
    kinematic_trust_coupling_loss,
    lwr_residual_per_sample,
    lwr_trust_coupling_loss,
    train_detector,
    evaluate_detector,
)
from piad_v2x.data_utils.feature_extractor import FEATURE_NAMES


def _zero_features(n: int) -> torch.Tensor:
    return torch.zeros(n, len(FEATURE_NAMES), dtype=torch.float32)


def _idx(name: str) -> int:
    return FEATURE_NAMES.index(name)


class Architecture(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        cfg = DetectorConfig(n_classes=20)
        model = PINNDetector(cfg)
        x = _zero_features(7)
        logits, trust = model(x)
        self.assertEqual(logits.shape, (7, 20))
        self.assertEqual(trust.shape, (7,))

    def test_trust_in_unit_interval(self) -> None:
        cfg = DetectorConfig()
        model = PINNDetector(cfg)
        x = torch.randn(32, len(FEATURE_NAMES))
        _, trust = model(x)
        self.assertTrue(torch.all(trust >= 0.0).item())
        self.assertTrue(torch.all(trust <= 1.0).item())


class KinematicViolation(unittest.TestCase):
    def test_zero_when_residuals_under_tolerance(self) -> None:
        cfg = DetectorConfig(tau_v=0.5, tau_a=4.0, tau_r=0.6)
        x = _zero_features(4)
        x[:, _idx("e_v")] = 0.1
        x[:, _idx("e_a")] = 1.0
        x[:, _idx("r_star")] = 0.3
        v = kinematic_violation(x, cfg)
        self.assertTrue(torch.all(v == 0.0).item())

    def test_positive_when_residuals_exceed_tolerance(self) -> None:
        cfg = DetectorConfig(tau_v=0.5, tau_a=4.0, tau_r=0.6)
        x = _zero_features(4)
        x[:, _idx("e_v")] = 10.0
        v = kinematic_violation(x, cfg)
        self.assertTrue(torch.all(v > 0.0).item())

    def test_violation_bounded_at_one(self) -> None:
        """tanh-squashed; saturates at 1.0 for extreme inputs."""
        cfg = DetectorConfig(tau_v=0.5, tau_a=4.0, tau_r=0.6)
        x = _zero_features(2)
        x[:, _idx("e_v")] = 1e6
        x[:, _idx("e_a")] = 1e6
        x[:, _idx("r_star")] = 1e6
        v = kinematic_violation(x, cfg)
        self.assertTrue(torch.all(v <= 1.0).item())
        self.assertTrue(torch.all(v >= 0.0).item())

    def test_cold_samples_excluded(self) -> None:
        cfg = DetectorConfig(tau_v=0.5, tau_a=4.0, tau_r=0.6)
        x = _zero_features(2)
        x[:, _idx("e_v")] = 100.0
        x[:, _idx("cold")] = 1.0
        v = kinematic_violation(x, cfg)
        self.assertTrue(torch.all(v == 0.0).item())

    def test_trust_coupling_gradient_only_through_trust(self) -> None:
        trust = torch.tensor([0.9, 0.5, 0.1], requires_grad=True)
        violation = torch.tensor([0.8, 0.3, 0.0])
        loss = kinematic_trust_coupling_loss(trust, violation)
        loss.backward()
        expected = violation / violation.numel()
        self.assertTrue(torch.allclose(trust.grad, expected))


class LWRResidual(unittest.TestCase):
    def test_zero_when_single_time_bin(self) -> None:
        cfg = DetectorConfig(lwr_bin_seconds=1.0)
        spd = torch.tensor([10.0, 11.0, 12.0])
        t = torch.tensor([0.1, 0.2, 0.3])
        r = lwr_residual_per_sample(spd, t, cfg)
        self.assertTrue(torch.all(r == 0.0).item())

    def test_zero_in_steady_state(self) -> None:
        """Constant count and constant flow across bins => residual ~ 0."""
        cfg = DetectorConfig(lwr_bin_seconds=1.0)
        # Bin 0 has 3 samples; bin 1 has 3 samples; same speed.
        spd = torch.tensor([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        t = torch.tensor([0.1, 0.2, 0.3, 1.1, 1.2, 1.3])
        r = lwr_residual_per_sample(spd, t, cfg)
        self.assertAlmostEqual(float(r.max()), 0.0, places=6)

    def test_positive_when_density_jumps(self) -> None:
        """3 vehicles in bin 0, 10 in bin 1 => big Δρ => non-zero residual."""
        cfg = DetectorConfig(lwr_bin_seconds=1.0)
        spd = torch.tensor([10.0] * 3 + [10.0] * 10)
        t = torch.tensor([0.1, 0.2, 0.3] + [1.0 + 0.05 * i for i in range(10)])
        r = lwr_residual_per_sample(spd, t, cfg)
        self.assertGreater(float(r.max()), 0.0)

    def test_trust_coupling_loss_gradient_only_through_trust(self) -> None:
        trust = torch.tensor([0.9, 0.5, 0.1], requires_grad=True)
        residual = torch.tensor([5.0, 1.0, 0.0])
        loss = lwr_trust_coupling_loss(trust, residual)
        loss.backward()
        # Gradient should be the residual divided by N (mean).
        expected = residual / residual.numel()
        self.assertTrue(torch.allclose(trust.grad, expected))


class SmallTrainingRun(unittest.TestCase):
    def test_two_epoch_run_reduces_loss(self) -> None:
        # Synthetic 2-class problem: class label determined by sign of first feature.
        rng = np.random.default_rng(0)
        n = 1024
        X = rng.normal(size=(n, len(FEATURE_NAMES))).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)
        send_time = rng.uniform(0.0, 10.0, size=n).astype(np.float64)
        cfg = DetectorConfig(
            n_classes=2,
            epochs=3,
            batch_size=128,
            lambda_kin=0.0,
            lambda_lwr=0.0,
            class_weighted=False,
        )
        _, result = train_detector(X, y, send_time, cfg=cfg, device=torch.device("cpu"))
        self.assertLess(result.history[-1]["loss_total"], result.history[0]["loss_total"])

    def test_inference_shape(self) -> None:
        rng = np.random.default_rng(1)
        n = 256
        X = rng.normal(size=(n, len(FEATURE_NAMES))).astype(np.float32)
        y = rng.integers(0, 3, size=n).astype(np.int64)
        send_time = rng.uniform(0.0, 5.0, size=n).astype(np.float64)
        cfg = DetectorConfig(n_classes=3, epochs=1, batch_size=64,
                             class_weighted=False)
        model, _ = train_detector(X, y, send_time, cfg=cfg,
                                  device=torch.device("cpu"))
        preds = evaluate_detector(model, X, y, device=torch.device("cpu"))
        self.assertEqual(preds.shape, (n,))
        self.assertTrue(set(preds.tolist()).issubset({0, 1, 2}))


class PhysicsConstrainedHead(unittest.TestCase):
    """Candidate D: physics embedded in the architecture (DD-009)."""

    def test_predictions_within_feasible_envelope(self) -> None:
        # The bounded kin head cannot emit a physically impossible state,
        # no matter how extreme the input.
        cfg = DetectorConfig()
        model = PINNDetector(cfg).eval()
        x = torch.randn(64, len(FEATURE_NAMES)) * 50.0
        _, _, (v_pred, a_pred, r_pred), _, _ = model.forward_with_physics(x)
        self.assertTrue(torch.all((v_pred >= 0) & (v_pred <= cfg.phys_v_max)).item())
        self.assertTrue(torch.all((a_pred >= 0) & (a_pred <= cfg.phys_a_max)).item())
        self.assertTrue(torch.all((r_pred >= 0) & (r_pred <= cfg.phys_r_max)).item())

    def test_impossible_state_raises_residual(self) -> None:
        cfg = DetectorConfig()
        model = PINNDetector(cfg).eval()
        feasible = _zero_features(1)
        feasible[:, _idx("spdx")] = 10.0
        impossible = _zero_features(1)
        impossible[:, _idx("spdx")] = 1.0e4   # physically impossible speed
        *_, r_feasible = model.forward_with_physics(feasible)
        *_, r_impossible = model.forward_with_physics(impossible)
        self.assertGreater(float(r_impossible), float(r_feasible))

    def test_physics_gradient_reaches_kin_head_and_encoder(self) -> None:
        # No detach: a classification-only loss must backprop through the
        # residual path into the kin head and the shared encoder.
        cfg = DetectorConfig(n_classes=3)
        model = PINNDetector(cfg)
        # Moderate input: keep the residual off its tanh/sigmoid saturation so
        # the gradient through the physics path is measurable (it exists at any
        # input; it merely vanishes once the residual saturates at an extreme).
        x = torch.randn(16, len(FEATURE_NAMES))
        logits, _, _, _, _ = model.forward_with_physics(x)
        logits.sum().backward()
        self.assertIsNotNone(model.kin_head.weight.grad)
        self.assertGreater(float(model.kin_head.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.encoder[0].weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
