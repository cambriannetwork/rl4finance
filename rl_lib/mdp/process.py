"""
Markov Decision Process classes.

This module provides classes for representing Markov Decision Processes (MDPs),
which are mathematical frameworks for modeling decision-making in situations
where outcomes are partly random and partly under the control of a decision maker.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Iterable, Tuple

from rl_lib.distribution.base import Distribution
from rl_lib.mdp.state import State, NonTerminal

# Type variables for state and action
S = TypeVar('S')
A = TypeVar('A')

class MarkovDecisionProcess(ABC, Generic[S, A]):
    """
    Base class for Markov Decision Processes.
    
    A Markov Decision Process (MDP) is defined by:
    - A set of states S
    - A set of actions A
    - A transition function P(s'|s,a) that gives the probability of transitioning
      to state s' when taking action a in state s
    - A reward function R(s,a,s') that gives the reward for transitioning from
      state s to state s' by taking action a
    
    This abstract class defines the interface for all MDPs in the RL library.
    """
    
    @abstractmethod
    def actions(self, state: NonTerminal[S]) -> Iterable[A]:
        """
        Return the available actions in the given state.
        
        Args:
            state: The current state
            
        Returns:
            Iterable of available actions
        """
        pass
    
    @abstractmethod
    def step(self, state: NonTerminal[S], action: A) -> Distribution[Tuple[State[S], float]]:
        """
        Take an action in a state and return a distribution over (next_state, reward) pairs.
        
        This method implements the transition and reward functions of the MDP.
        
        Args:
            state: The current state
            action: The action to take
            
        Returns:
            Distribution over (next_state, reward) pairs
        """
        pass
    
    def is_terminal(self, state: State[S]) -> bool:
        """
        Check if a state is terminal.
        
        Args:
            state: The state to check
            
        Returns:
            True if the state is terminal, False otherwise
        """
        return not isinstance(state, NonTerminal)
