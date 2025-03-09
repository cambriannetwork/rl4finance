"""
Base classes for function approximation.

This module provides abstract base classes for function approximation,
which is used to approximate value functions and policies in RL.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, Iterable, Tuple, Sequence, Any

import numpy as np

# Type variables
X = TypeVar('X')  # Input type for function approximation
F = TypeVar('F', bound='FunctionApprox')  # Function approximation type

class FunctionApprox(ABC, Generic[X]):
    """
    Interface for function approximations.
    
    This abstract class represents a function approximation that maps inputs of type X
    to real numbers. It can be evaluated at specific points and updated with
    additional (X, ℝ) points.
    """
    
    @abstractmethod
    def __add__(self: F, other: F) -> F:
        """
        Add two function approximations.
        
        Args:
            other: Another function approximation of the same type
            
        Returns:
            New function approximation representing the sum
        """
        pass
    
    @abstractmethod
    def __mul__(self: F, scalar: float) -> F:
        """
        Multiply a function approximation by a scalar.
        
        Args:
            scalar: A scalar value
            
        Returns:
            New function approximation representing the product
        """
        pass
    
    @abstractmethod
    def objective_gradient(
        self: F,
        xy_vals_seq: Iterable[Tuple[X, float]],
        obj_deriv_out_fun: Callable[[Sequence[X], Sequence[float]], np.ndarray]
    ) -> 'Gradient[F]':
        """
        Compute the gradient of an objective function with respect to parameters.
        
        This method computes the gradient of an objective function of the self
        FunctionApprox with respect to the parameters in the internal
        representation of the FunctionApprox.
        
        Args:
            xy_vals_seq: Sequence of (x, y) pairs
            obj_deriv_out_fun: Function to compute derivative of objective
            
        Returns:
            Gradient object
        """
        pass
    
    @abstractmethod
    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        """
        Compute expected value of y for each x in x_values_seq.
        
        Args:
            x_values_seq: Sequence of x values
            
        Returns:
            Array of predicted y values
        """
        pass
    
    def __call__(self, x_value: X) -> float:
        """
        Evaluate the function at a single point.
        
        Args:
            x_value: Input value
            
        Returns:
            Predicted output value
        """
        return self.evaluate([x_value]).item()
    
    @abstractmethod
    def update_with_gradient(
        self: F,
        gradient: 'Gradient[F]'
    ) -> F:
        """
        Update the internal parameters using the input gradient.
        
        Args:
            gradient: Gradient object
            
        Returns:
            Updated function approximation
        """
        pass
    
    def update(
        self: F,
        xy_vals_seq: Iterable[Tuple[X, float]]
    ) -> F:
        """
        Update parameters based on (x, y) pairs.
        
        This method updates the internal parameters of the function approximation
        based on incremental data provided in the form of (x, y) pairs.
        
        Args:
            xy_vals_seq: Sequence of (x, y) pairs
            
        Returns:
            Updated function approximation
        """
        def deriv_func(x: Sequence[X], y: Sequence[float]) -> np.ndarray:
            """Compute the derivative of the objective function."""
            return self.evaluate(x) - np.array(y)
        
        return self.update_with_gradient(
            self.objective_gradient(xy_vals_seq, deriv_func)
        )
    
    @abstractmethod
    def solve(
        self: F,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: float = None
    ) -> F:
        """
        Fit parameters to the given data.
        
        This method assumes the entire data set of (x, y) pairs is available
        and solves for the internal parameters of the function approximation
        such that they are fitted to the data.
        
        Args:
            xy_vals_seq: Sequence of (x, y) pairs
            error_tolerance: Optional error tolerance for convergence
            
        Returns:
            Fitted function approximation
        """
        pass
    
    @abstractmethod
    def within(self: F, other: F, tolerance: float) -> bool:
        """
        Check if this function approximation is within tolerance of another.
        
        Args:
            other: Another function approximation of the same type
            tolerance: Tolerance for comparison
            
        Returns:
            True if within tolerance, False otherwise
        """
        pass


@dataclass(frozen=True)
class Gradient(Generic[F]):
    """
    Gradient of a FunctionApprox with respect to its parameters.
    
    This class represents the gradient of a function approximation with respect
    to its internal parameters. It is used for gradient-based optimization.
    """
    
    function_approx: F
    """Function approximation representing the gradient"""
    
    def __add__(self, other: 'Gradient[F]') -> 'Gradient[F]':
        """
        Add two gradients.
        
        Args:
            other: Another gradient of the same type
            
        Returns:
            New gradient representing the sum
        """
        return Gradient(self.function_approx + other.function_approx)
    
    def __mul__(self, scalar: float) -> 'Gradient[F]':
        """
        Multiply a gradient by a scalar.
        
        Args:
            scalar: A scalar value
            
        Returns:
            New gradient representing the product
        """
        return Gradient(self.function_approx * scalar)
