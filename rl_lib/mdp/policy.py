"""
Policy classes for Markov Decision Processes.

This module provides classes for representing policies in Markov Decision Processes.
A policy defines the behavior of an agent by mapping states to distributions over actions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable

from rl_lib.distribution.base import Distribution
from rl_lib.distribution.discrete import Constant
from rl_lib.mdp.state import NonTerminal

# Type variables for state and action
S = TypeVar('S')
A = TypeVar('A')

class Policy(ABC, Generic[S, A]):
    """
    A policy maps states to distributions over actions.
    
    A policy defines the behavior of an agent in a Markov Decision Process.
    It specifies what action to take in each state, possibly stochastically.
    """
    
    @abstractmethod
    def act(self, state: NonTerminal[S]) -> Distribution[A]:
        """
        Return a distribution over actions for the given state.
        
        Args:
            state: The current state
            
        Returns:
            Distribution over actions
        """
        pass
    
    def __call__(self, state: NonTerminal[S]) -> Distribution[A]:
        """
        Convenience method to call act.
        
        Args:
            state: The current state
            
        Returns:
            Distribution over actions
        """
        return self.act(state)


@dataclass(frozen=True)
class DeterministicPolicy(Policy[S, A]):
    """
    A policy that returns a single action for each state.
    
    A deterministic policy always selects the same action in a given state.
    It is represented as a function from states to actions.
    """
    
    action_for: Callable[[S], A]
    """Function that maps state values to actions"""
    
    def act(self, state: NonTerminal[S]) -> Constant[A]:
        """
        Return a constant distribution with the determined action.
        
        Args:
            state: The current state
            
        Returns:
            Constant distribution with the selected action
        """
        action = self.action_for(state.state)
        return Constant(action)
    
    def __repr__(self) -> str:
        """
        Return a string representation of the policy.
        
        Returns:
            String representation
        """
        return f"DeterministicPolicy({self.action_for})"
