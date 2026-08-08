"""YAML configuration loader.

Central entry point for reading the structured experiment configurations under
`piad_v2x/hypes_yaml/`. Every tunable hyperparameter of the detector and the
trust-gated pseudonym lifecycle is declared in a YAML file and loaded into a
configuration dictionary, so that no hyperparameter is hardcoded in the tools.
"""
from __future__ import annotations

import os

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_yaml(path: str) -> dict:
    """Load a configuration file into a nested dictionary.

    A bare filename (for example ``multiscale_rf.yaml``) is resolved against the
    ``piad_v2x/hypes_yaml`` directory; an explicit path is used as given.
    """
    if not os.path.isabs(path) and not os.path.exists(path):
        candidate = os.path.join(_HERE, path)
        if os.path.exists(candidate):
            path = candidate
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
