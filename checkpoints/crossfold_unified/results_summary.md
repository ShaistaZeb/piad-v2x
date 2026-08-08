# PIAD-V2X Cross-Fold / Multi-Seed Evaluation

Seeds: [17, 23, 42]  |  folds: 5  |  successful runs: 15  |  rows: 200,000  |  holdout: [1, 3, 5, 7, 9]  |  PINN epochs: 6  |  primary operating point: 5% FPR  |  device: cpu

## Observed evidence

Mean ± SD [95% CI] across all runs (threshold chosen on calibration slice, metrics on disjoint evaluation slice).

| Model | AUC | Recall @5% FPR | F1 @5% FPR | Accuracy @5% FPR | Recall @1% | Recall @10% |
|---|---|---|---|---|---|---|
| Random Forest | 0.8266 ± 0.0124 [0.8198, 0.8335] | 0.4670 ± 0.0189 [0.4565, 0.4774] | 0.5568 ± 0.0203 [0.5456, 0.5680] | 0.8620 ± 0.0073 [0.8580, 0.8660] | 0.4166 ± 0.0128 [0.4095, 0.4237] | 0.5137 ± 0.0195 [0.5029, 0.5244] |
| PINN-kin | 0.8233 ± 0.0121 [0.8166, 0.8300] | 0.5300 ± 0.0322 [0.5122, 0.5479] | 0.6070 ± 0.0301 [0.5903, 0.6236] | 0.8729 ± 0.0053 [0.8700, 0.8759] | 0.4130 ± 0.0332 [0.3946, 0.4314] | 0.5979 ± 0.0242 [0.5845, 0.6113] |
| PINN-full | 0.8257 ± 0.0103 [0.8200, 0.8314] | 0.5286 ± 0.0303 [0.5118, 0.5453] | 0.6056 ± 0.0283 [0.5899, 0.6213] | 0.8725 ± 0.0058 [0.8692, 0.8757] | 0.4113 ± 0.0307 [0.3943, 0.4283] | 0.6087 ± 0.0214 [0.5968, 0.6205] |

### Paired statistical tests (same seeds and folds)

Paired t-test and Wilcoxon signed-rank, n=15 per pair. Bonferroni alpha for 3 comparisons = 0.0167.

| Metric | Comparison | Mean diff (A-B) | t p-value | Wilcoxon p | Significant (Bonferroni) |
|---|---|---|---|---|---|
| auc | Random Forest vs PINN-kin | 0.0033 | 0.3870 | 0.5614 | no |
| auc | Random Forest vs PINN-full | 0.0009 | 0.8026 | 0.6387 | no |
| auc | PINN-kin vs PINN-full | -0.0024 | 0.0684 | 0.0637 | no |
| recall_fpr5 | Random Forest vs PINN-kin | -0.0631 | 0.0000 | 0.0001 | YES |
| recall_fpr5 | Random Forest vs PINN-full | -0.0616 | 0.0000 | 0.0001 | YES |
| recall_fpr5 | PINN-kin vs PINN-full | 0.0014 | 0.7796 | 0.8904 | no |
| f1 | Random Forest vs PINN-kin | -0.0501 | 0.0000 | 0.0003 | YES |
| f1 | Random Forest vs PINN-full | -0.0488 | 0.0000 | 0.0001 | YES |
| f1 | PINN-kin vs PINN-full | 0.0014 | 0.7519 | 0.8469 | no |

## Interpretation

- Random Forest vs PINN-kin (AUC): Random Forest higher by 0.0033, but p=0.3870 is NOT significant after Bonferroni; no superiority claim is warranted.
- Random Forest vs PINN-full (AUC): Random Forest higher by 0.0009, but p=0.8026 is NOT significant after Bonferroni; no superiority claim is warranted.
- PINN-kin vs PINN-full (AUC): PINN-full higher by 0.0024, but p=0.0684 is NOT significant after Bonferroni; no superiority claim is warranted.

Superiority is asserted only where a paired test is significant after correction; elsewhere the wording is deliberately 'no difference established'.

## Limitations

- The 5 folds within each seed share the same subsampled pool, so per-run results are not fully independent. The paired tests are therefore approximate and likely anti-conservative (real p-values may be larger).
- Only 3 seeds x 5 folds; a larger grid would tighten the CIs.
- One dataset (VeReMi Extension), one held-out attack family set ([1, 3, 5, 7, 9]), one feature set; results may not transfer.
- Operating points use a fixed 5% (and 1/10%) benign FPR target; a different deployment FPR would shift the recall/precision trade-off.
- CPU training, fixed PINN hyperparameters (lambda, epochs); a PINN hyperparameter search was not performed, so 'PINN does not beat RF' is conditional on these settings.
