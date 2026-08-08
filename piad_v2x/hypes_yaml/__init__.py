"""Central experiment configurations and the YAML loader.

Each configuration file declares the full parameter set for one study; the tools
under ``piad_v2x/tools`` consume them through :func:`load_yaml`.
"""
from .yaml_utils import load_yaml

__all__ = ["load_yaml"]
