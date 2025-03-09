"""
Function approximation module for RL library.

This module provides classes for function approximation, which is used
to approximate value functions and policies in reinforcement learning.
"""

from rl_lib.function_approx.base import FunctionApprox, Gradient
from rl_lib.function_approx.gradient import AdamGradient
from rl_lib.function_approx.weights import Weights
from rl_lib.function_approx.dnn import DNNSpec, DNNApprox

__all__ = [
    'FunctionApprox',
    'Gradient',
    'AdamGradient',
    'Weights',
    'DNNSpec',
    'DNNApprox'
]
