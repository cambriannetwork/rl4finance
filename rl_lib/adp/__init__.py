"""
Approximate Dynamic Programming (ADP) module for RL library.

This module provides functions for solving Markov Decision Processes
using Approximate Dynamic Programming techniques.
"""

from rl_lib.adp.backward import (
    extended_vf,
    back_opt_vf_and_policy,
    back_opt_qvf,
    ValueFunctionApprox,
    QValueFunctionApprox,
    NTStateDistribution
)

__all__ = [
    'extended_vf',
    'back_opt_vf_and_policy',
    'back_opt_qvf',
    'ValueFunctionApprox',
    'QValueFunctionApprox',
    'NTStateDistribution'
]
