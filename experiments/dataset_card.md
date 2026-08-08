# Dataset Card - VeReMi Extension features (PIAD-V2X)

Real raw VeReMi Extension (CC-BY 4.0), low-density `_1416` windows, features from `piad_v2x/tools/veremi_extract.py`. Rows = received BSMs after dropping unresolved senders (attackerType = -1).

| Scenario | Messages | Senders | Attacker senders | Attacker msg frac | med r_pos_cv benign | med r_pos_cv attacker |
|---|---|---|---|---|---|---|
| ConstPos_1416 | 763,999 | 1,688 | 507 | 0.311 | 0.21 | 11.72 |
| ConstSpeed_1416 | 763,999 | 1,688 | 507 | 0.311 | 0.2 | 32.98 |
| DataReplaySybil_1416 | 763,999 | 1,688 | 507 | 0.311 | 0.2 | 0.88 |
| DataReplay_1416 | 763,999 | 1,688 | 507 | 0.311 | 0.21 | 13.54 |
| DelayedMessages_1416 | 763,999 | 1,688 | 507 | 0.311 | 0.2 | 0.0 |
| Disruptive_1416 | 763,999 | 1,688 | 507 | 0.311 | 0.2 | 227.53 |
| DoS_1416 | 1,234,670 | 1,688 | 507 | 0.574 | 0.2 | 3.3 |
| EventualStop_1416 | 763,999 | 1,688 | 507 | 0.311 | 0.2 | 0.0 |
| RandomPos_1416 | 763,999 | 1,688 | 507 | 0.311 | 0.2 | 770.1 |
| RandomSpeed_1416 | 763,999 | 1,688 | 507 | 0.311 | 0.2 | 32.16 |

**Totals:** 8,110,661 messages across 16,880 sender-instances in 10 scenarios.

Sender-instances are the group unit for the leakage-safe split (60/20/20). The effective N for sender-clustered inference is the sender count, not the message count.
