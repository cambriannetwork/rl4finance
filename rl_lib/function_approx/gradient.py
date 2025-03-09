"""
Gradient-based optimization for function approximation.

This module provides classes for gradient-based optimization of function approximations,
including the Adam optimizer.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class AdamGradient:
    """
    Parameters for Adam gradient descent optimizer.
    
    Adam (Adaptive Moment Estimation) is an optimization algorithm that can be used
    instead of the classical stochastic gradient descent procedure to update network
    weights in a more efficient way.
    
    References:
        Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
        arXiv preprint arXiv:1412.6980.
    """
    
    learning_rate: float
    """Learning rate (step size)"""
    
    decay1: float
    """Exponential decay rate for the first moment estimates"""
    
    decay2: float
    """Exponential decay rate for the second moment estimates"""
    
    @staticmethod
    def default_settings() -> 'AdamGradient':
        """
        Return default settings for Adam optimizer.
        
        Returns:
            AdamGradient with default settings
        """
        return AdamGradient(
            learning_rate=0.001,
            decay1=0.9,
            decay2=0.999
        )
    
    def __repr__(self) -> str:
        """
        Return a string representation of the Adam gradient.
        
        Returns:
            String representation
        """
        return (f"AdamGradient(learning_rate={self.learning_rate}, "
                f"decay1={self.decay1}, decay2={self.decay2})")
