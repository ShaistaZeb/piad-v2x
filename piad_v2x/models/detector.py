"""Multi-physics PINN detector.

The physics-informed neural detector, with the must-have losses:

    L = L_cls + λ_kin · L_kin + λ_lwr · L_lwr

L_proto and L_trust are spec'd but parked behind feature flags so the
smoke run stays minimal.

Architecture (per spec):
    19 kinematic features
        -> MLP encoder (3 hidden, ReLU): 19 -> 128 -> 64 -> 64
            -> classification head: 64 -> 20 logits (softmax at inference)
            -> trust head: 64 -> 1, sigmoid -> [0, 1]

L_kin   : hinge-style loss on per-sample kinematic residuals, gated to 0 on
          cold-start rows. Tolerance bands from vehicle-dynamics norms
          (τ_v = 0.5 m/s, τ_a = 4.0 m/s², τ_r = 0.6 rad/s).

L_lwr   : Lighthill-Whitham-Richards conservation residual per time bin in
          the batch, multiplied by the model's trust score. Encodes "if the
          local neighbourhood violates mass conservation, trust should fall."
          The residual is detached so gradients flow only through s_t.
          Anchor precedent: Thodi et al. 2024 (PI-FNO with LWR for traffic
          state estimation in J. Computational Physics).

Design intent: this is the FIRST trained PINN. The smoke run in
`experiments/run_pinn.py` confirms forward + backward both work and that the
multi-physics ablation is one parameter away. The held-out-attack
comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..data_utils.feature_extractor import FEATURE_NAMES


# Index lookups against the feature vector (see feature_extractor.FEATURE_NAMES).
_F = {name: i for i, name in enumerate(FEATURE_NAMES)}


@dataclass(frozen=True)
class DetectorConfig:
    # Architecture
    input_dim: int = len(FEATURE_NAMES)
    hidden: tuple[int, ...] = (128, 64, 64)
    n_classes: int = 20
    dropout: float = 0.1

    # Loss weights
    lambda_kin: float = 0.5
    lambda_lwr: float = 0.5
    # L_jerk: microscopic acceleration-magnitude penalty (added 2026-06-10
    # as a forward direction motivated by the lambda sweep finding).
    lambda_jerk: float = 0.0
    tau_jerk: float = 5.0   # hinge threshold (m/s^2)

    # L_kin tolerance bands.
    tau_v: float = 0.5
    tau_a: float = 4.0
    tau_r: float = 0.6

    # L_lwr binning
    lwr_bin_seconds: float = 1.0
    lwr_residual_scale: float = 20.0   # divisor so residual sits ~O(1) per bin

    # --- in-architecture physics-constrained kinematic head ---
    # The kin head predicts a feasible (speed, accel, turn-rate) state through
    # bounded activations, so the network cannot emit a physically impossible
    # kinematic state. The reconstruction residual is computed inside forward()
    # and structurally feeds the trust score and the benign-class logit; the
    # reconstruction loss is un-detached, so physics shapes the shared encoder.
    phys_v_max: float = 70.0          # m/s   feasible speed envelope
    phys_a_max: float = 10.0          # m/s^2 feasible acceleration envelope
    phys_r_max: float = 1.5           # rad/s feasible turn-rate envelope
    physics_trust_gain: float = 2.0   # how strongly the residual suppresses trust
    physics_logit_gain: float = 1.0   # how strongly the residual lowers the benign logit
    lambda_recon: float = 1.0         # weight of the un-detached benign reconstruction loss

    # Optimisation
    batch_size: int = 512
    epochs: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-5
    class_weighted: bool = True
    seed: int = 17

    # Numerical stability
    normalise_inputs: bool = True
    grad_clip: float = 5.0


class PINNDetector(nn.Module):
    """Encoder + classification head + trust head."""

    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = cfg.input_dim
        for w in cfg.hidden:
            layers += [nn.Linear(prev, w), nn.ReLU(), nn.Dropout(cfg.dropout)]
            prev = w
        self.encoder = nn.Sequential(*layers)
        self.cls_head = nn.Linear(prev, cfg.n_classes)
        self.trust_head = nn.Linear(prev, 1)
        # Physics-constrained kinematic head: predicts (v, a, r) bounded to the
        # feasible envelope via sigmoid, so the model cannot represent an
        # impossible kinematic state (a hard constraint).
        self.kin_head = nn.Linear(prev, 3)
        self.v_max, self.a_max, self.r_max = cfg.phys_v_max, cfg.phys_a_max, cfg.phys_r_max
        self.beta, self.gamma = cfg.physics_trust_gain, cfg.physics_logit_gain
        # Input normalisation buffers (mean, std). Filled at training time;
        # default identity makes the model usable without calibration.
        self.register_buffer("input_mean", torch.zeros(cfg.input_dim))
        self.register_buffer("input_std", torch.ones(cfg.input_dim))
        # One-hot selector for the benign class (index 0); used to bias logits.
        _benign = torch.zeros(cfg.n_classes)
        _benign[0] = 1.0
        self.register_buffer("benign_mask", _benign)

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.input_mean) / self.input_std

    def _observed_kinematics(self, x: torch.Tensor):
        """Observed physical kinematics from the RAW (un-normalised) features."""
        eps = 1e-12
        obs_v = torch.sqrt(x[:, _F["spdx"]] ** 2 + x[:, _F["spdy"]] ** 2 + eps)
        obs_a = torch.sqrt(x[:, _F["aclx"]] ** 2 + x[:, _F["acly"]] ** 2 + eps)
        obs_r = x[:, _F["r_star"]].abs()
        return obs_v, obs_a, obs_r

    def forward_with_physics(self, x: torch.Tensor):
        """Forward pass exposing the physics terms (for the reconstruction loss).

        The kinematic head emits a feasible-by-construction (v, a, r); the
        normalised residual against the observed kinematics is squashed to [0, 1)
        and structurally (a) suppresses the trust score and (b) lowers the
        benign-class logit. Nothing is detached, so the residual path trains the
        shared encoder as well as the kin head.
        """
        z = self.encoder(self.normalise(x))
        kp = self.kin_head(z)
        v_pred = self.v_max * torch.sigmoid(kp[:, 0])
        a_pred = self.a_max * torch.sigmoid(kp[:, 1])
        r_pred = self.r_max * torch.sigmoid(kp[:, 2])
        obs_v, obs_a, obs_r = self._observed_kinematics(x)
        rv = (v_pred - obs_v) / self.v_max
        ra = (a_pred - obs_a) / self.a_max
        rr = (r_pred - obs_r) / self.r_max
        residual = torch.tanh(rv * rv + ra * ra + rr * rr)   # in [0, 1)
        trust = torch.sigmoid(self.trust_head(z).squeeze(-1) - self.beta * residual)
        logits = self.cls_head(z) - self.gamma * residual.unsqueeze(1) * self.benign_mask
        return logits, trust, (v_pred, a_pred, r_pred), (obs_v, obs_a, obs_r), residual

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, trust, _, _, _ = self.forward_with_physics(x)
        return logits, trust


# ----------------------- physics losses -----------------------

def kinematic_violation(
    x: torch.Tensor,
    cfg: DetectorConfig,
) -> torch.Tensor:
    """Per-sample kinematic violation indicator in [0, 1).

    Computes the spec's hinge form, normalised by each tolerance band so the
    three terms are comparable, then squashed through tanh so a single
    teleport-attack outlier cannot dominate batch statistics. Cold-start
    samples are masked to 0.
    """
    e_v = x[:, _F["e_v"]]
    e_a = x[:, _F["e_a"]]
    r_star = x[:, _F["r_star"]]
    cold = x[:, _F["cold"]]
    warm = 1.0 - cold

    h_v = torch.clamp(e_v.abs() / cfg.tau_v - 1.0, min=0.0)
    h_a = torch.clamp(e_a.abs() / cfg.tau_a - 1.0, min=0.0)
    h_r = torch.clamp(r_star.abs() / cfg.tau_r - 1.0, min=0.0)

    return torch.tanh(h_v + h_a + h_r) * warm


def jerk_magnitude_violation(
    x: torch.Tensor,
    cfg: DetectorConfig,
) -> torch.Tensor:
    """Per-sample microscopic-jerk-magnitude violation indicator.

    Penalises samples whose implied acceleration magnitude a_star
    exceeds a realistic-driving bound tau_jerk (default 5 m/s^2).
    Cold-start samples are masked. Returns values in [0, 1] via tanh.
    """
    a_star = x[:, _F["a_star"]]
    cold = x[:, _F["cold"]]
    warm = 1.0 - cold
    h = torch.clamp(a_star.abs() / cfg.tau_jerk - 1.0, min=0.0)
    return torch.tanh(h) * warm


def kinematic_trust_coupling_loss(
    trust: torch.Tensor,
    violation: torch.Tensor,
) -> torch.Tensor:
    """L_kin: trust score is penalised in proportion to kinematic violation.

    Implements the PINN-soft-constraint pattern (mirrors L_lwr): when the
    sample's kinematic residual exceeds the physical tolerance band, trust
    should fall. The violation is detached so gradient flows only through
    the trust head, then back to the shared encoder.
    """
    return (trust * violation.detach()).mean()


def lwr_residual_per_sample(
    spd_mag: torch.Tensor,
    send_time: torch.Tensor,
    cfg: DetectorConfig,
) -> torch.Tensor:
    """Compute per-sample LWR conservation residual from the batch.

    Bin samples by sendTime into windows of cfg.lwr_bin_seconds. For each
    consecutive bin pair (b, b+1), compute the discrete continuity residual:

        r_b = (ρ_{b+1} - ρ_b) / Δt + (q_{b+1} - q_b)

    where ρ_b is sample count in bin b and q_b = ρ_b · mean_speed_b.

    Each sample inherits the absolute residual of its time bin. Returns a
    tensor of shape (n,) in approximate [0, 1] after dividing by
    cfg.lwr_residual_scale.
    """
    if spd_mag.numel() == 0:
        return torch.zeros_like(spd_mag)

    bins = torch.floor(send_time / cfg.lwr_bin_seconds).long()
    unique_bins, inverse = torch.unique(bins, sorted=True, return_inverse=True)
    n_bins = int(unique_bins.numel())
    if n_bins < 2:
        return torch.zeros_like(spd_mag)

    device = spd_mag.device
    dtype = spd_mag.dtype
    rho = torch.zeros(n_bins, device=device, dtype=dtype)
    rho.scatter_add_(0, inverse, torch.ones_like(spd_mag))
    spd_sum = torch.zeros(n_bins, device=device, dtype=dtype)
    spd_sum.scatter_add_(0, inverse, spd_mag)
    mean_spd = spd_sum / rho.clamp_min(1.0)
    q = rho * mean_spd

    bin_residual = torch.zeros(n_bins, device=device, dtype=dtype)
    bin_residual[:-1] = (rho[1:] - rho[:-1]) / cfg.lwr_bin_seconds + (q[1:] - q[:-1])
    per_sample = bin_residual.abs()[inverse] / cfg.lwr_residual_scale
    return per_sample.clamp(0.0, 5.0)


def lwr_trust_coupling_loss(
    trust: torch.Tensor,
    residual_per_sample: torch.Tensor,
) -> torch.Tensor:
    """L_lwr: trust score is penalised in proportion to local LWR violation.

    Implements the spec intent: "Sybil clusters violate conservation; this
    is the partner physics prior to L_kin." Residual is detached so the
    gradient flows only through the trust head.
    """
    return (trust * residual_per_sample.detach()).mean()


# ----------------------- training -----------------------

@dataclass(frozen=True)
class TrainResult:
    history: list[dict] = field(default_factory=list)
    final_loss: float = 0.0
    device: str = "cpu"


def _pick_device(prefer_mps: bool = False) -> torch.device:
    """Pick the best device.

    MPS is opt-in: in PyTorch 2.12 we observed NaN losses on this model
    architecture under MPS even with vanilla CE, while CPU and CUDA work.
    Until a future PyTorch fixes the MPS path for our op mix, default to CPU.
    """
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_detector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    send_time_train: np.ndarray,
    cfg: DetectorConfig | None = None,
    device: torch.device | None = None,
    use_amp: bool = False,
) -> tuple[PINNDetector, TrainResult]:
    """Train the PINN. `send_time_train` carries the raw sendTime per row,
    needed to bin samples for L_lwr. Automatic mixed precision (`use_amp`) is
    engaged only on CUDA devices; on CPU it is inert and training proceeds in
    full precision."""
    cfg = cfg or DetectorConfig()
    device = device or _pick_device()
    amp_on = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    X_t = torch.as_tensor(np.ascontiguousarray(X_train), dtype=torch.float32)
    y_t = torch.as_tensor(np.ascontiguousarray(y_train), dtype=torch.long)
    st_t = torch.as_tensor(np.ascontiguousarray(send_time_train), dtype=torch.float32)

    ds = TensorDataset(X_t, y_t, st_t)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    model = PINNDetector(cfg).to(device)
    if cfg.normalise_inputs:
        # Per-feature mean/std on the training set; std floor avoids div-by-0
        # on constant features (e.g. cold flag is 0/1).
        mu = X_t.mean(dim=0)
        sigma = X_t.std(dim=0).clamp_min(1e-6)
        model.input_mean.copy_(mu.to(device))
        model.input_std.copy_(sigma.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.class_weighted:
        counts = np.bincount(y_train, minlength=cfg.n_classes).astype(np.float64)
        # Inverse-frequency weighting; clamp tiny classes to a sane floor.
        weights = 1.0 / np.maximum(counts, 1.0)
        weights = weights / weights.mean()
        cls_w = torch.as_tensor(weights, dtype=torch.float32, device=device)
        ce = nn.CrossEntropyLoss(weight=cls_w)
    else:
        ce = nn.CrossEntropyLoss()

    result = TrainResult(device=str(device))
    for epoch in range(cfg.epochs):
        model.train()
        running = {"cls": 0.0, "recon": 0.0, "lwr": 0.0, "total": 0.0, "n_batches": 0}
        for xb, yb, stb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            stb = stb.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=amp_on):
                logits, trust, (v_pred, a_pred, r_pred), (obs_v, obs_a, obs_r), _ = \
                    model.forward_with_physics(xb)
            l_cls = ce(logits, yb)
            # Un-detached reconstruction on benign rows: the kin head learns the
            # feasible benign manifold, and physics gradients reach the encoder.
            benign = yb == 0
            if benign.any():
                rv = (v_pred[benign] - obs_v[benign]) / cfg.phys_v_max
                ra = (a_pred[benign] - obs_a[benign]) / cfg.phys_a_max
                rr = (r_pred[benign] - obs_r[benign]) / cfg.phys_r_max
                l_recon = (rv * rv + ra * ra + rr * rr).mean()
            else:
                l_recon = torch.zeros((), device=device)
            # Optional neighbourhood-density auxiliary (LWR); off unless enabled.
            if cfg.lambda_lwr > 0.0:
                r_lwr = lwr_residual_per_sample(
                    spd_mag=torch.sqrt(xb[:, _F["spdx"]] ** 2 + xb[:, _F["spdy"]] ** 2),
                    send_time=stb,
                    cfg=cfg,
                )
                l_lwr = lwr_trust_coupling_loss(trust, r_lwr)
            else:
                l_lwr = torch.zeros((), device=device)

            loss = l_cls + cfg.lambda_recon * l_recon + cfg.lambda_lwr * l_lwr
            opt.zero_grad()
            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0.0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            running["cls"] += float(l_cls.detach())
            running["recon"] += float(l_recon.detach())
            running["lwr"] += float(l_lwr.detach())
            running["total"] += float(loss.detach())
            running["n_batches"] += 1

        nb = max(1, running["n_batches"])
        epoch_log = {
            "epoch": epoch + 1,
            "loss_cls": running["cls"] / nb,
            "loss_recon": running["recon"] / nb,
            "loss_lwr": running["lwr"] / nb,
            "loss_total": running["total"] / nb,
        }
        result.history.append(epoch_log)

    object.__setattr__(result, "final_loss", result.history[-1]["loss_total"] if result.history else 0.0)
    return model, result


@torch.no_grad()
def evaluate_detector(
    model: PINNDetector,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device | None = None,
    batch_size: int = 4096,
) -> np.ndarray:
    """Return predicted class labels for X_test."""
    device = device or _pick_device()
    model.eval()
    X_t = torch.as_tensor(np.ascontiguousarray(X_test), dtype=torch.float32)
    preds: list[np.ndarray] = []
    for i in range(0, len(X_t), batch_size):
        xb = X_t[i:i + batch_size].to(device)
        logits, _ = model(xb)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def predict_proba(
    model: PINNDetector,
    X_test: np.ndarray,
    device: torch.device | None = None,
    batch_size: int = 4096,
) -> np.ndarray:
    """Return softmax probabilities over classes for X_test. Shape (n, n_classes)."""
    device = device or _pick_device()
    model.eval()
    X_t = torch.as_tensor(np.ascontiguousarray(X_test), dtype=torch.float32)
    chunks: list[np.ndarray] = []
    for i in range(0, len(X_t), batch_size):
        xb = X_t[i:i + batch_size].to(device)
        logits, _ = model(xb)
        chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0)
