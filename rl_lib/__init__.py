"""
Reinforcement Learning Library.

This library provides components for building and solving reinforcement learning problems,
including Markov Decision Processes, function approximation, and dynamic programming.
"""

__version__ = '0.1.0'

# Import submodules to make them available through the package
from rl_lib import distribution
from rl_lib import mdp
from rl_lib import function_approx
from rl_lib import utils
from rl_lib import adp

__all__ = [
    'distribution',
    'mdp',
    'function_approx',
    'utils',
    'adp'
]
