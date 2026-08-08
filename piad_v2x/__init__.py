"""PIAD-V2X: Physics-Informed Adaptive Defence for V2X.

Reference implementation of a multi-scale physics misbehaviour detector coupled
to a trust-gated pseudonym lifecycle.
"""
from .lifecycle.coupling import (
    CouplingConfig,
    CouplingLayer,
    NeighbourhoodSummary,
    PeerState,
)
from .lifecycle.aggregator import AggregatorConfig, TrustAggregator

__all__ = [
    "CouplingConfig",
    "CouplingLayer",
    "NeighbourhoodSummary",
    "PeerState",
    "AggregatorConfig",
    "TrustAggregator",
]
