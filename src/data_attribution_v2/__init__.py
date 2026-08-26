"""Correctness-gated experiments for local-to-multi-round data selection."""

from .f2sa import PenaltySchedule, f2sa_coordinate_gradient, vf_score
from .omega import OmegaState, budgeted_sigmoid, capped_simplex_projection
from .protocol import Method, ProtocolSpec, selection_update_epochs

__all__ = [
    "Method",
    "OmegaState",
    "PenaltySchedule",
    "ProtocolSpec",
    "budgeted_sigmoid",
    "capped_simplex_projection",
    "f2sa_coordinate_gradient",
    "selection_update_epochs",
    "vf_score",
]
