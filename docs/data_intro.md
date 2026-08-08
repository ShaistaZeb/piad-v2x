# Data Introduction

PIAD-V2X is evaluated on the public **VeReMi Extension** benchmark (Kamel et al.,
2020; CC-BY 4.0). This document describes the raw data, the features the detector
consumes, and the attack scenarios. Download and extraction commands are in
[`../data/README.md`](../data/README.md).

## Raw format

The benchmark ships one nested ZIP per scenario. Inside, each receiving vehicle has
a JSON log of the messages it heard. Three message types matter:

- **type 2** - the receiver's own GPS fix (ego state).
- **type 3** - a received safety message (BSM), carrying `sendTime`, `sender` (the
  true vehicle id, used only as an oracle label), `senderPseudo` (the pseudonym the
  sender used), `messageID`, and the claimed position, speed, acceleration, heading.
- **type 4** - the ground-truth transmitted state per `messageID`, used to measure
  attack magnitude for the adaptive-attacker study.

`piad_v2x/tools/veremi_extract.py` streams these logs and emits one row per received
message into a per-scenario parquet under `data/features/`.

## Labels and the leakage-safe split

Each vehicle's attacker type is encoded in its trace filename (`A0` = genuine,
`A1..` = attacker). A message is labelled by its true sender's type (`is_attacker`,
`attackerType`). `sender` and the label are **oracle** columns: the detector never
sees them; they are used only for labelling and for grouping.

Train/test are split by bare `sender` (60/20/40), so every message from one vehicle
stays in one partition even after pseudonym rotation. The membership is pinned in
`experiments/results/split_map.json` and consumed by every script.

## Features

Two feature representations are produced from the same raw extraction:

- **Multi-scale detector** (the deployed model): 16 single-vehicle kinematic
  features (position, speed, acceleration, heading, displacement, the
  constant-velocity residual `r_pos_cv`, the speed/position residual `r_spd_pos`,
  and so on) plus 4 multi-vehicle interaction invariants computed per one-second
  scene - `I1` mutual occupancy, `I2` reachability, `I3` collision feasibility. The
  invariants require the states of neighbouring vehicles and are the part an
  attacker cannot forge. See `piad_v2x/models/relational.py`.
- **Loss-based detector** (the physics-in-loss comparison): a 24-element vector
  defined once in `piad_v2x/data_utils/feature_extractor.py::FEATURE_NAMES` - 8 raw
  fields, 5 kinematic deltas, 3 inferred quantities, 2 consistency residuals, a
  cold-start mask, and 5 plausibility flags.

## Scenarios

Each attack family is provided in a sparse (`_1416`) and a dense (`_0709`) traffic
window:

| Scenario | Attack |
|---|---|
| ConstPos | constant position offset |
| RandomPos | random position offset |
| ConstSpeed / RandomSpeed | speed falsification |
| DataReplay | replay of a genuine earlier message |
| DataReplaySybil | replay across many pseudonyms (Sybil) |
| DelayedMessages | genuine but stale messages |
| Disruptive | erratic falsification |
| DoS | flooding |
| EventualStop | benign until a sudden stop |

The corpus is about 8.1 million received messages across roughly 1,700 sender
vehicles. Per-scenario counts are in `experiments/dataset_card.md`.

A second, independent benchmark - **VeReMi NextGen** (InTAS / Ingolstadt) - is used
for the external-validity check via `piad_v2x/tools/nextgen_extract.py`.
