"""
Backward induction algorithms for Approximate Dynamic Programming.

This module provides functions for solving Markov Decision Processes
using backward induction with function approximation.
"""

from typing import TypeVar, Tuple, Sequence, Iterator, Callable, Dict, Any
from operator import itemgetter
import numpy as np

from rl_lib.distribution.base import Distribution
from rl_lib.mdp.state import State, NonTerminal
from rl_lib.mdp.process import MarkovDecisionProcess
from rl_lib.mdp.policy import Policy, DeterministicPolicy
from rl_lib.function_approx.base import FunctionApprox
from rl_lib.logging import get_logger, log_phase, log_convergence, log_weights_summary

# Type variables
S = TypeVar('S')  # State type
A = TypeVar('A')  # Action type

# Type aliases for clarity
ValueFunctionApprox = FunctionApprox[NonTerminal[S]]  # For approximating V(s)
QValueFunctionApprox = FunctionApprox[Tuple[NonTerminal[S], A]]  # For approximating Q(s,a)
NTStateDistribution = Distribution[NonTerminal[S]]  # Distribution over non-terminal states

# Triple of (MDP, initial function approximation, state distribution)
MDP_FuncApproxV_Distribution = Tuple[
    MarkovDecisionProcess[S, A],
    ValueFunctionApprox[S],
    NTStateDistribution[S]
]

MDP_FuncApproxQ_Distribution = Tuple[
    MarkovDecisionProcess[S, A],
    QValueFunctionApprox[S, A],
    NTStateDistribution[S]
]


def extended_vf(vf: ValueFunctionApprox[S], s: State[S]) -> float:
    """
    Evaluate value function, returning 0 for terminal states.
    
    Args:
        vf: Value function approximation
        s: State to evaluate
        
    Returns:
        Value of state, or 0 if terminal
    """
    if isinstance(s, NonTerminal):
        return vf(s)
    else:
        return 0.0


def back_opt_vf_and_policy(
    mdp_f0_mu_triples: Sequence[MDP_FuncApproxV_Distribution[S, A]],
    γ: float,
    num_state_samples: int,
    error_tolerance: float
) -> Iterator[Tuple[ValueFunctionApprox[S], DeterministicPolicy[S, A]]]:
    """
    Use backwards induction to find the optimal value function and policy.
    
    This function uses backward induction to find the optimal value function
    and policy at each time step, using function approximation for the value function.
    
    Args:
        mdp_f0_mu_triples: Sequence of (MDP, initial function approx, state distribution)
        γ: Discount factor
        num_state_samples: Number of states to sample at each time step
        error_tolerance: Error tolerance for convergence
        
    Returns:
        Iterator of (value function, policy) pairs for each time step
    """
    # Get logger
    logger = get_logger()
    
    # Log the start of backward induction
    log_phase("back_opt_vf_and_policy_start", {
        "horizon": len(mdp_f0_mu_triples),
        "discount_factor": γ,
        "num_state_samples": num_state_samples,
        "error_tolerance": error_tolerance
    })
    
    # Get the number of time steps
    horizon = len(mdp_f0_mu_triples)
    
    # Initialize list to store value functions and policies
    vp = []
    
    # Process time steps in reverse order
    for i, (mdp, approx0, mu) in enumerate(reversed(mdp_f0_mu_triples)):
        # Log the start of processing for this time step
        logger.info({
            "event": "processing_time_step_vf",
            "time_step": horizon - i - 1,  # Convert to forward time index
            "reverse_index": i
        })
        
        # Define return function for this time step
        def return_(s_r: Tuple[State[S], float], i=i) -> float:
            """Compute return for a (state, reward) pair."""
            s1, r = s_r
            # Add discounted value of next state if not at first time step
            return r + γ * (extended_vf(vp[i-1][0], s1) if i > 0 else 0.)
        
        # Sample states from the distribution
        sampled_states = mu.sample_n(num_state_samples)
        
        # Log state sampling
        logger.debug({
            "event": "sampled_states_vf",
            "time_step": horizon - i - 1,
            "num_states": len(sampled_states),
            "sample_stats": {
                "min": float(min(s.state for s in sampled_states)) if sampled_states else None,
                "max": float(max(s.state for s in sampled_states)) if sampled_states else None,
                "mean": float(sum(s.state for s in sampled_states) / len(sampled_states)) if sampled_states else None
            }
        })
        
        # For each sampled state, compute the optimal value
        training_data = []
        for s in sampled_states:
            # Find the action with maximum expected return
            max_val = float('-inf')
            for a in mdp.actions(s):
                # Compute expected return for this action
                exp_return = mdp.step(s, a).expectation(return_)
                max_val = max(max_val, exp_return)
            
            # Add (state, optimal value) pair to training data
            training_data.append((s, max_val))
        
        # Log training data statistics
        if training_data:
            v_vals = [v for _, v in training_data]
            logger.debug({
                "event": "training_data_stats_vf",
                "time_step": horizon - i - 1,
                "num_samples": len(training_data),
                "value_stats": {
                    "min": float(min(v_vals)),
                    "max": float(max(v_vals)),
                    "mean": float(sum(v_vals) / len(v_vals)),
                    "std": float(np.std(v_vals)) if len(v_vals) > 1 else 0.0
                }
            })
        
        # Solve for the value function approximation
        this_v = approx0.solve(training_data, error_tolerance)
        
        # Log weights summary
        log_weights_summary(this_v.weights, f"vf_weights_time_{horizon - i - 1}")
        
        # Define the deterministic policy
        def deter_policy(state: S) -> A:
            """Return the optimal action for a state."""
            nt_state = NonTerminal(state)
            
            # Find action with maximum expected return
            best_action = None
            best_return = float('-inf')
            
            for a in mdp.actions(nt_state):
                exp_return = mdp.step(nt_state, a).expectation(return_)
                if exp_return > best_return:
                    best_return = exp_return
                    best_action = a
            
            return best_action
        
        # Create policy
        policy = DeterministicPolicy(deter_policy)
        
        # Add value function and policy to the list
        vp.append((this_v, policy))
        
        # Log completion of this time step
        logger.info({
            "event": "time_step_completed_vf",
            "time_step": horizon - i - 1,
            "reverse_index": i,
            "remaining_steps": i
        })
    
    # Log completion of backward induction
    log_phase("back_opt_vf_and_policy_complete", {
        "horizon": horizon,
        "num_value_functions": len(vp)
    })
    
    # Return value functions and policies in forward order
    return reversed(vp)


def back_opt_qvf(
    mdp_f0_mu_triples: Sequence[MDP_FuncApproxQ_Distribution[S, A]],
    γ: float,
    num_state_samples: int,
    error_tolerance: float
) -> Iterator[QValueFunctionApprox[S, A]]:
    """
    Use backwards induction to find the optimal Q-value function.
    
    This function uses backward induction to find the optimal Q-value function
    at each time step, using function approximation for the Q-value function.
    
    Args:
        mdp_f0_mu_triples: Sequence of (MDP, initial Q-function approx, state distribution)
        γ: Discount factor
        num_state_samples: Number of states to sample at each time step
        error_tolerance: Error tolerance for convergence
        
    Returns:
        Iterator of Q-value functions for each time step
    """
    # Get logger
    logger = get_logger()
    
    # Log the start of backward induction
    log_phase("back_opt_qvf_start", {
        "horizon": len(mdp_f0_mu_triples),
        "discount_factor": γ,
        "num_state_samples": num_state_samples,
        "error_tolerance": error_tolerance
    })
    
    # Get the number of time steps
    horizon = len(mdp_f0_mu_triples)
    
    # Initialize list to store Q-value functions
    qvf = []
    
    # Process time steps in reverse order
    for i, (mdp, approx0, mu) in enumerate(reversed(mdp_f0_mu_triples)):
        # Log the start of processing for this time step
        logger.info({
            "event": "processing_time_step",
            "time_step": horizon - i - 1,  # Convert to forward time index
            "reverse_index": i
        })
        
        # Define return function for this time step
        def return_(s_r: Tuple[State[S], float], i=i) -> float:
            """Compute return for a (state, reward) pair."""
            s1, r = s_r
            
            # Add discounted maximum Q-value of next state if not at first time step
            next_return = 0.0
            if i > 0 and isinstance(s1, NonTerminal):
                next_mdp = mdp_f0_mu_triples[horizon - i][0]
                # Find maximum Q-value for next state
                max_q = float('-inf')
                for a in next_mdp.actions(s1):
                    q_val = qvf[i-1]((s1, a))
                    max_q = max(max_q, q_val)
                next_return = max_q
            
            return r + γ * next_return
        
        # Sample states from the distribution
        sampled_states = mu.sample_n(num_state_samples)
        
        # Log state sampling
        logger.debug({
            "event": "sampled_states",
            "time_step": horizon - i - 1,
            "num_states": len(sampled_states),
            "sample_stats": {
                "min": float(min(s.state for s in sampled_states)) if sampled_states else None,
                "max": float(max(s.state for s in sampled_states)) if sampled_states else None,
                "mean": float(sum(s.state for s in sampled_states) / len(sampled_states)) if sampled_states else None
            }
        })
        
        # For each sampled state and action, compute the Q-value
        training_data = []
        for s in sampled_states:
            for a in mdp.actions(s):
                # Compute expected return for this state-action pair
                q_val = mdp.step(s, a).expectation(return_)
                training_data.append(((s, a), q_val))
        
        # Log training data statistics
        if training_data:
            q_vals = [q for _, q in training_data]
            logger.debug({
                "event": "training_data_stats",
                "time_step": horizon - i - 1,
                "num_samples": len(training_data),
                "q_value_stats": {
                    "min": float(min(q_vals)),
                    "max": float(max(q_vals)),
                    "mean": float(sum(q_vals) / len(q_vals)),
                    "std": float(np.std(q_vals)) if len(q_vals) > 1 else 0.0
                }
            })
        
        # Solve for the Q-value function approximation
        this_qvf = approx0.solve(training_data, error_tolerance)
        
        # Log weights summary
        log_weights_summary(this_qvf.weights, f"qvf_weights_time_{horizon - i - 1}")
        
        # Add Q-value function to the list
        qvf.append(this_qvf)
        
        # Log completion of this time step
        logger.info({
            "event": "time_step_completed",
            "time_step": horizon - i - 1,
            "reverse_index": i,
            "remaining_steps": i
        })
    
    # Log completion of backward induction
    log_phase("back_opt_qvf_complete", {
        "horizon": horizon,
        "num_q_functions": len(qvf)
    })
    
    # Return Q-value functions in forward order
    return reversed(qvf)
