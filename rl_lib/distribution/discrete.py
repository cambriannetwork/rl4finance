"""
Discrete probability distributions.
"""

import random
from dataclasses import dataclass
from typing import List, Callable, TypeVar, Generic, Iterable, Any

from rl_lib.distribution.base import Distribution

# Type variable for distribution outcomes
T = TypeVar('T')

class Choose(Distribution[T]):
    """
    Uniform distribution over a finite set of options.
    
    This class represents a discrete uniform distribution over a given set of options.
    Each option has equal probability of being selected.
    """
    
    def __init__(self, options: Iterable[T]):
        """
        Initialize a uniform choice distribution.
        
        Args:
            options: Collection of items to choose from with equal probability
        """
        self.options = list(options)
        
        if not self.options:
            raise ValueError("Options list cannot be empty")
    
    def sample(self) -> T:
        """
        Return a random sample from this distribution.
        
        Returns:
            A randomly chosen option
        """
        return random.choice(self.options)
    
    def expectation(self, f: Callable[[T], float]) -> float:
        """
        Calculate expectation by averaging over all possible outcomes.
        
        For a discrete uniform distribution, the expectation is the average
        of the function applied to each possible outcome.
        
        Args:
            f: Function to apply to each outcome
            
        Returns:
            Expected value of f(X)
        """
        total = 0.0
        for option in self.options:
            total += f(option)
        return total / len(self.options)
    
    def __repr__(self) -> str:
        """
        Return a string representation of the distribution.
        
        Returns:
            String representation
        """
        if len(self.options) <= 5:
            options_str = str(self.options)
        else:
            options_str = f"[{', '.join(str(o) for o in self.options[:3])}, ..., {self.options[-1]}]"
        return f"Choose({options_str})"


@dataclass(frozen=True)
class Constant(Distribution[T]):
    """
    A distribution with a single outcome that has probability 1.
    
    This class represents a degenerate distribution that always returns the same value.
    It is useful for representing deterministic outcomes in stochastic contexts.
    """
    value: T
    
    def sample(self) -> T:
        """
        Return the constant value.
        
        Returns:
            The constant value
        """
        return self.value
    
    def expectation(self, f: Callable[[T], float]) -> float:
        """
        Return f(value) since the distribution is a point mass.
        
        For a constant distribution, the expectation of f(X) is simply f(value).
        
        Args:
            f: Function to apply to the outcome
            
        Returns:
            f(value)
        """
        return f(self.value)
    
    def __repr__(self) -> str:
        """
        Return a string representation of the distribution.
        
        Returns:
            String representation
        """
        return f"Constant({self.value})"
