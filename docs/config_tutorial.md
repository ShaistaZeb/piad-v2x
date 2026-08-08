# Configuration

Every tunable hyperparameter of the detector and the trust-gated pseudonym
lifecycle is declared in a YAML file under `piad_v2x/hypes_yaml/` and loaded into a
configuration dictionary through `piad_v2x.hypes_yaml.load_yaml`. No hyperparameter
is hardcoded on the execution path. The lifecycle constants were frozen before the
closed-loop runs and the operating point is
pre-registered.

## Configuration files

| File | Study |
|---|---|
| `piad_v2x/hypes_yaml/multiscale_rf.yaml` | The deployed multi-scale detector coupled to the trust-gated lifecycle (the headline result). |
| `piad_v2x/hypes_yaml/pinn_placement.yaml` | The detector placement study (data-only / residual-as-feature / residual-in-loss) and the LWR conservation-loss ablation. |

Loading a configuration:

```python
from piad_v2x.hypes_yaml import load_yaml
cfg = load_yaml("multiscale_rf.yaml")   # a bare name resolves against hypes_yaml/
```

The tools accept the file through `--hypes_yaml`:

```bash
python piad_v2x/tools/run_lifecycle_multiscale.py --hypes_yaml multiscale_rf.yaml
```

## Detector (the deployed model)

Declared under `detector:` in `multiscale_rf.yaml`:

| Key | Value |
|---|---|
| `core_method` | `random_forest` |
| `feature_set` | kinematic (16) + interaction invariants (4) |
| `args.n_estimators` | 120 |
| `args.max_depth` | 22 |
| `args.class_weight` | balanced |
| `calibration.target_recall` | 0.90 (decision threshold set on validation) |

The depth-22 value is frozen, not tuned on the test set; a sender-disjoint depth
sweep confirms deeper trees do not improve generalisation.

## Trust-gated lifecycle

Declared under `trust_gate:`:

| Key | Value | Meaning |
|---|---|---|
| `alpha` | 0.8 | EWMA weight on prior trust |
| `t0` | 0.5 | neutral initial trust |
| `n_min` | 5 | cold-start messages before the gate can fire |
| `tau_revoke` | 0.15 | revoke when trust stays below this |
| `k_consecutive` | 3 | consecutive updates below the threshold before revocation |
| `base_cadence_s` | 60.0 | base pseudonym rotation cadence |

The pre-registered operating point is `tau_revoke = 0.15, k_consecutive = 3`. Only
the gate threshold `tau_revoke` was tuned (on a training window); the remaining
constants are fixed a priori.

## Neural comparators (placement study)

`pinn_placement.yaml` declares the physics-informed neural detector under
`detector.args` (two hidden layers of 128 and 64 units, `lambda_kin`, `lambda_lwr`),
the optimizer (`adam`, learning rate 1e-3), and the training schedule. Setting
`lambda_lwr: 0.0` selects the kinematic-only variant. These configurations back the
pre-registered negative result that physics-in-loss does not improve detection.

## Changing a parameter

Edit the relevant YAML file and re-run the tool with `--hypes_yaml`. The regression
tests in `tests/test_headline_results.py` pin the published numbers, so a change
that moves a headline value fails the suite.
