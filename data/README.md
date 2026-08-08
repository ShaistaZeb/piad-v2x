# Data (one pipeline; only the raw dataset is external)

This repository is self-contained except for the raw dataset, which is public and
downloaded once. Both detector phases derive from the SAME raw extraction - there is no second
dataset.

## Pipeline

```
raw VeReMi Extension scenario zips  (download below)  ->  data/raw/
  piad_v2x/tools/veremi_extract.py     ->  data/features/*.parquet   (multi-scale detector uses directly)
  piad_v2x/tools/build_mixed_table.py ->  data/veremi.parquet       (loss-based detector mixed table)
```

## Step 1 - download the raw dataset

VeReMi Extension (Kamel et al., 2020; CC-BY 4.0). Download the per-scenario zips into
`data/raw/`:

- https://mega.nz/folder/z0pnGA4a#WFEUISyS5_maabhcEI7HQA
  (linked from https://github.com/josephkamel/VeReMi-Dataset)

Or point at an existing local copy without moving files:

    export PIAD_VEREMI_RAW=/path/to/your/VeReMi/zips

Each scenario arrives as a folder of window zips. Repack each into a single
outer zip named after the scenario, stored not recompressed:

    cd data/raw && for d in */; do s="${d%/}"; (cd "$s" && zip -0 -q -r "../$s.zip" .); done

## Step 2 - extract per-scenario features

    python piad_v2x/tools/veremi_extract.py \
        --scenario DataReplay_1416 --out data/features/DataReplay_1416.parquet
    # repeat per scenario, or script the loop

## Step 3 - build loss-based detector mixed table

    python piad_v2x/tools/build_mixed_table.py --out data/veremi.parquet

The loss-based detector then runs with `--data data/veremi.parquet` (the loader reads parquet natively).

Everything under `data/` (`raw/`, `features/`, `*.parquet`) is git-ignored; only these
instructions are tracked.
