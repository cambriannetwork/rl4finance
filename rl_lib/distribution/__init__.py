"""
Distribution module for RL library.

This module provides classes for representing probability distributions
that can be sampled and used to compute expectations.
"""

from rl_lib.distribution.base import Distribution
from rl_lib.distribution.sampled import SampledDistribution, Gaussian
from rl_lib.distribution.discrete import Choose, Constant

__all__ = [
    'Distribution',
    'SampledDistribution',
    'Gaussian',
    'Choose',
    'Constant'
]
