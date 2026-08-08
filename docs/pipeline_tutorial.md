# Pipeline Tutorial

End-to-end walkthrough from the raw dataset to the headline result. Run every
command from the repository root with the virtual environment active. Data setup is
in [`../data/README.md`](../data/README.md); parameters are in
[`config_tutorial.md`](config_tutorial.md).

## 1. Extract features

Turn the raw VeReMi Extension zips into per-scenario feature tables:

```bash
python piad_v2x/tools/veremi_extract.py --scenario DataReplay_1416 \
    --out data/features/DataReplay_1416.parquet
# repeat per scenario, or script the loop
python piad_v2x/tools/build_mixed_table.py --out data/veremi.parquet
```

`veremi_extract.py` writes the multi-scale feature parquet; `build_mixed_table.py`
joins them into the single table the loss-based detector reads.

## 2. Train the detector

```bash
# placement study: Random Forest + two physics-in-loss neural variants
python piad_v2x/tools/train.py --data data/veremi.parquet --rows 200000 --out checkpoints/
# detector ablation (the five-way comparison behind Table 5.1)
python piad_v2x/tools/train_detector.py
```

Models and a `manifest.json` (which records every parameter needed to rebuild the
exact split) are written to `checkpoints/` and `experiments/results/models/`.

## 3. Evaluate

```bash
python piad_v2x/tools/inference.py --ckpt checkpoints/     # inference only, never retrains
python piad_v2x/tools/crossfold.py --data data/veremi.parquet --out checkpoints/   # 3 seeds x 5 folds
```

`crossfold.py` produces the paired statistics behind the "physics-in-loss is inert"
result.

## 4. Run the closed loop

```bash
# per-attack acceptance under no defence / detection-only / full coupling
python -m piad_v2x.lifecycle.closed_loop
# the headline: trains the multi-scale detector on _1416, scores held-out _0709
python piad_v2x/tools/run_lifecycle_multiscale.py
# the second outcome metric, with censoring guards
python piad_v2x/tools/run_time_to_isolate.py
```

`run_lifecycle_multiscale.py` writes `experiments/results/lifecycle_multiscale.json`
- the 95.6% reduction / 0.20% false-revocation / 99.6% isolated / 1.0 s result.

## 5. Demonstrate one attacker

```bash
python piad_v2x/tools/demo_trace.py
```

Replays one attacker pseudonym message by message and shows its trust decaying
across the gate to revocation, with the count of malicious messages accepted before
isolation.

## 6. Tests

```bash
python -m unittest discover -s tests
```

`tests/test_headline_results.py` asserts the committed numbers, so a pipeline change
that alters a published claim fails here.
