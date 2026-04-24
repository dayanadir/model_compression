"""Selector registry."""

from compression.selectors.activation import ActivationSelector
from compression.selectors.in_out_meanabs import InOutMeanAbsSelector
from compression.selectors.l1_structured import L1StructuredSelector
from compression.selectors.magnitude import MagnitudeSelector
from compression.selectors.random_consistent import RandomConsistentSelector
from compression.selectors.uniform import UniformSelector

__all__ = [
    "ActivationSelector",
    "InOutMeanAbsSelector",
    "L1StructuredSelector",
    "MagnitudeSelector",
    "RandomConsistentSelector",
    "UniformSelector",
]
