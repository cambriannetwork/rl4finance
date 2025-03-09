"""
Weight management for neural networks.

This module provides classes for managing weights in neural networks,
including support for Adam optimization.
"""

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from rl_lib.function_approx.gradient import AdamGradient

# Small number to avoid division by zero
SMALL_NUM = 1e-6

@dataclass(frozen=True)
class Weights:
    """
    Weights for a neural network with Adam optimizer state.
    
    This class represents the weights of a neural network layer, along with
    the state of the Adam optimizer used to update them.
    """
    
    adam_gradient: AdamGradient
    """Parameters for Adam optimizer"""
    
    time: int
    """Number of updates performed"""
    
    weights: np.ndarray
    """Weight matrix"""
    
    adam_cache1: np.ndarray
    """First moment vector (biased)"""
    
    adam_cache2: np.ndarray
    """Second moment vector (biased)"""
    
    @staticmethod
    def create(
        weights: np.ndarray,
        adam_gradient: Optional[AdamGradient] = None,
        adam_cache1: Optional[np.ndarray] = None,
        adam_cache2: Optional[np.ndarray] = None
    ) -> 'Weights':
        """
        Create a new Weights object with optional Adam optimizer state.
        
        Args:
            weights: Weight matrix
            adam_gradient: Optional Adam optimizer parameters
            adam_cache1: Optional first moment vector
            adam_cache2: Optional second moment vector
            
        Returns:
            New Weights object
        """
        # Use default Adam parameters if not provided
        if adam_gradient is None:
            adam_gradient = AdamGradient.default_settings()
        
        # Initialize Adam caches to zeros if not provided
        if adam_cache1 is None:
            adam_cache1 = np.zeros_like(weights)
        
        if adam_cache2 is None:
            adam_cache2 = np.zeros_like(weights)
        
        return Weights(
            adam_gradient=adam_gradient,
            time=0,
            weights=weights,
            adam_cache1=adam_cache1,
            adam_cache2=adam_cache2
        )
    
    def update(self, gradient: np.ndarray) -> 'Weights':
        """
        Update weights using Adam optimizer.
        
        This method implements the Adam optimization algorithm to update
        the weights based on the provided gradient.
        
        Args:
            gradient: Gradient of the objective function with respect to weights
            
        Returns:
            New Weights object with updated weights and Adam state
        """
        # Increment time step
        time = self.time + 1
        
        # Update biased first moment estimate
        new_adam_cache1 = (
            self.adam_gradient.decay1 * self.adam_cache1 +
            (1 - self.adam_gradient.decay1) * gradient
        )
        
        # Update biased second moment estimate
        new_adam_cache2 = (
            self.adam_gradient.decay2 * self.adam_cache2 +
            (1 - self.adam_gradient.decay2) * gradient ** 2
        )
        
        # Compute bias-corrected first moment estimate
        corrected_m = new_adam_cache1 / (1 - self.adam_gradient.decay1 ** time)
        
        # Compute bias-corrected second moment estimate
        corrected_v = new_adam_cache2 / (1 - self.adam_gradient.decay2 ** time)
        
        # Update weights
        new_weights = (
            self.weights -
            self.adam_gradient.learning_rate * corrected_m /
            (np.sqrt(corrected_v) + SMALL_NUM)
        )
        
        # Return new Weights object with updated state
        return replace(
            self,
            time=time,
            weights=new_weights,
            adam_cache1=new_adam_cache1,
            adam_cache2=new_adam_cache2,
        )
    
    def within(self, other: 'Weights', tolerance: float) -> bool:
        """
        Check if weights are within tolerance of another set of weights.
        
        Args:
            other: Another Weights object
            tolerance: Tolerance for comparison
            
        Returns:
            True if all weights are within tolerance, False otherwise
        """
        return np.all(np.abs(self.weights - other.weights) <= tolerance).item()
    
    def __repr__(self) -> str:
        """
        Return a string representation of the weights.
        
        Returns:
            String representation
        """
        shape_str = 'x'.join(str(dim) for dim in self.weights.shape)
        return f"Weights(shape={shape_str}, time={self.time})"
