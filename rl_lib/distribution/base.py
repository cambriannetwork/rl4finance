"""
Base classes for probability distributions that can be sampled.
"""

from abc import ABC, abstractmethod
from typing import Callable, List, TypeVar, Generic

# Type variable for distribution outcomes
T = TypeVar('T')

class Distribution(ABC, Generic[T]):
    """
    Base class for probability distributions that can be sampled.
    
    This abstract class defines the interface for all probability distributions
    in the RL library. Distributions can be sampled and used to compute expectations.
    """
    
    @abstractmethod
    def sample(self) -> T:
        """
        Return a random sample from this distribution.
        
        Returns:
            A random outcome from the distribution
        """
        pass
    
    def sample_n(self, n: int) -> List[T]:
        """
        Return n samples from this distribution.
        
        Args:
            n: Number of samples to generate
            
        Returns:
            List of n random samples
        """
        return [self.sample() for _ in range(n)]
    
    @abstractmethod
    def expectation(self, f: Callable[[T], float]) -> float:
        """
        Return the expectation of f(X) where X is the random variable.
        
        Args:
            f: Function to apply to each outcome
            
        Returns:
            Expected value of f(X)
        """
        pass
