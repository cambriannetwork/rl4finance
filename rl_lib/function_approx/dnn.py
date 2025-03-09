"""
Neural network function approximation.

This module provides classes for neural network function approximation,
which can be used to approximate value functions and policies in RL.
"""

from dataclasses import dataclass, replace
from typing import Sequence, Callable, Iterable, Tuple, List, Optional, TypeVar

import numpy as np
import itertools

from rl_lib.function_approx.base import FunctionApprox, Gradient
from rl_lib.function_approx.gradient import AdamGradient
from rl_lib.function_approx.weights import Weights
from rl_lib.utils.iterate import converged, accumulate

# Type variable for input type
X = TypeVar('X')

@dataclass(frozen=True)
class DNNSpec:
    """
    Specification for a neural network architecture.
    
    This class defines the architecture of a neural network, including
    the number of neurons in each layer, whether to use bias terms,
    and the activation functions for hidden and output layers.
    """
    
    neurons: Sequence[int]
    """Number of neurons in each hidden layer"""
    
    bias: bool
    """Whether to include bias terms"""
    
    hidden_activation: Callable[[np.ndarray], np.ndarray]
    """Activation function for hidden layers"""
    
    hidden_activation_deriv: Callable[[np.ndarray], np.ndarray]
    """Derivative of hidden activation function"""
    
    output_activation: Callable[[np.ndarray], np.ndarray]
    """Activation function for output layer"""
    
    output_activation_deriv: Callable[[np.ndarray], np.ndarray]
    """Derivative of output activation function"""
    
    def __repr__(self) -> str:
        """
        Return a string representation of the DNN specification.
        
        Returns:
            String representation
        """
        return (f"DNNSpec(neurons={self.neurons}, bias={self.bias}, "
                f"hidden_activation={self.hidden_activation.__name__}, "
                f"output_activation={self.output_activation.__name__})")


@dataclass(frozen=True)
class DNNApprox(FunctionApprox[X]):
    """
    Neural network function approximation.
    
    This class implements a neural network function approximation that maps
    inputs of type X to real numbers. It can be used to approximate value
    functions and policies in reinforcement learning.
    """
    
    feature_functions: Sequence[Callable[[X], float]]
    """Functions to extract features from inputs"""
    
    dnn_spec: DNNSpec
    """Neural network architecture specification"""
    
    regularization_coeff: float
    """L2 regularization coefficient"""
    
    weights: Sequence[Weights]
    """Weights for each layer of the network"""
    
    @staticmethod
    def create(
        feature_functions: Sequence[Callable[[X], float]],
        dnn_spec: DNNSpec,
        adam_gradient: AdamGradient = None,
        regularization_coeff: float = 0.0,
        weights: Optional[Sequence[Weights]] = None
    ) -> 'DNNApprox[X]':
        """
        Create a new neural network function approximation.
        
        Args:
            feature_functions: Functions to extract features from inputs
            dnn_spec: Neural network architecture specification
            adam_gradient: Optional Adam optimizer parameters
            regularization_coeff: L2 regularization coefficient
            weights: Optional pre-initialized weights
            
        Returns:
            New DNNApprox instance
        """
        # Use default Adam parameters if not provided
        if adam_gradient is None:
            adam_gradient = AdamGradient.default_settings()
        
        # Initialize weights if not provided
        if weights is None:
            # Calculate input and output dimensions for each layer
            inputs = [len(feature_functions)] + [
                n + (1 if dnn_spec.bias else 0)
                for i, n in enumerate(dnn_spec.neurons)
            ]
            outputs = list(dnn_spec.neurons) + [1]
            
            # Initialize weights using Xavier/Glorot initialization
            wts = [
                Weights.create(
                    weights=np.random.randn(output, inp) / np.sqrt(inp),
                    adam_gradient=adam_gradient
                ) 
                for inp, output in zip(inputs, outputs)
            ]
        else:
            wts = weights
        
        return DNNApprox(
            feature_functions=feature_functions,
            dnn_spec=dnn_spec,
            regularization_coeff=regularization_coeff,
            weights=wts
        )
    
    def get_feature_values(self, x_values_seq: Iterable[X]) -> np.ndarray:
        """
        Extract features from input values.
        
        Args:
            x_values_seq: Sequence of input values
            
        Returns:
            Array of feature values for each input
        """
        return np.array([
            [f(x) for f in self.feature_functions]
            for x in x_values_seq
        ])
    
    def forward_propagation(self, x_values_seq: Iterable[X]) -> List[np.ndarray]:
        """
        Perform forward pass through the neural network.
        
        Args:
            x_values_seq: Sequence of input values
            
        Returns:
            List of activations for each layer, with the final output as the last element
        """
        # Extract features from inputs
        inp = self.get_feature_values(x_values_seq)
        activations = [inp]
        
        # Process hidden layers
        for i, w in enumerate(self.weights[:-1]):
            # Apply weights and activation function
            out = self.dnn_spec.hidden_activation(np.dot(inp, w.weights.T))
            
            # Add bias term if specified
            if self.dnn_spec.bias:
                inp = np.insert(out, 0, 1., axis=1)
            else:
                inp = out
            
            activations.append(inp)
        
        # Process output layer
        output = self.dnn_spec.output_activation(
            np.dot(inp, self.weights[-1].weights.T)
        )[:, 0]
        
        activations.append(output)
        
        return activations
    
    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        """
        Evaluate the network on input values.
        
        Args:
            x_values_seq: Sequence of input values
            
        Returns:
            Array of predicted values
        """
        return self.forward_propagation(x_values_seq)[-1]
    
    def backward_propagation(
        self,
        fwd_prop: Sequence[np.ndarray],
        obj_deriv_out: np.ndarray
    ) -> List[np.ndarray]:
        """
        Perform backpropagation to compute gradients.
        
        Args:
            fwd_prop: Result of forward propagation (activations)
            obj_deriv_out: Derivative of objective function with respect to output
            
        Returns:
            List of gradients for each layer's weights
        """
        # Reshape derivative to match output layer
        deriv = obj_deriv_out.reshape(1, -1)
        
        # Initialize list of gradients
        gradients = []
        
        # Check if fwd_prop has enough elements
        if len(fwd_prop) < 2:
            # Handle the case where there are no hidden layers
            gradients = [np.dot(deriv, fwd_prop[0]) / deriv.shape[1]]
        else:
            # Handle output layer
            gradients.append(np.dot(deriv, fwd_prop[-2]) / deriv.shape[1])
        
        # Backpropagate through hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            # Compute derivative with respect to layer output
            deriv = np.dot(self.weights[i + 1].weights.T, deriv)
            
            # Apply activation derivative
            if self.dnn_spec.bias:
                # For layers with bias, we need to handle the bias term separately
                deriv_with_activation = deriv * self.dnn_spec.hidden_activation_deriv(
                    fwd_prop[i + 1].T
                )
                # Remove bias term gradient
                deriv = deriv_with_activation[1:]
            else:
                deriv = deriv * self.dnn_spec.hidden_activation_deriv(
                    fwd_prop[i + 1].T
                )
            
            # Compute gradient for layer weights
            gradients.append(np.dot(deriv, fwd_prop[i]) / deriv.shape[1])
        
        # Reverse gradients to match layer order
        return gradients[::-1]
    
    def objective_gradient(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        obj_deriv_out_fun: Callable[[Sequence[X], Sequence[float]], np.ndarray]
    ) -> Gradient['DNNApprox[X]']:
        """
        Compute gradient of objective function with respect to network parameters.
        
        Args:
            xy_vals_seq: Sequence of (x, y) pairs
            obj_deriv_out_fun: Function to compute derivative of objective
            
        Returns:
            Gradient object
        """
        # Split inputs and outputs
        x_vals, y_vals = zip(*xy_vals_seq)
        
        # Compute derivative of objective function
        obj_deriv_out = obj_deriv_out_fun(x_vals, y_vals)
        
        # Perform forward and backward passes
        fwd_prop = self.forward_propagation(x_vals)[:-1]  # Exclude final output
        gradients = self.backward_propagation(fwd_prop, obj_deriv_out)
        
        # Add L2 regularization
        regularized_gradients = [
            g + self.regularization_coeff * self.weights[i].weights
            for i, g in enumerate(gradients)
        ]
        
        # Create gradient object
        return Gradient(replace(
            self,
            weights=[replace(w, weights=g) for w, g in zip(self.weights, regularized_gradients)]
        ))
    
    def __add__(self, other: 'DNNApprox[X]') -> 'DNNApprox[X]':
        """
        Add two neural networks by adding their weights.
        
        Args:
            other: Another neural network of the same architecture
            
        Returns:
            New neural network with summed weights
        """
        if not isinstance(other, DNNApprox):
            raise TypeError("Can only add DNNApprox instances")
        
        return replace(
            self,
            weights=[
                replace(w, weights=w.weights + o.weights)
                for w, o in zip(self.weights, other.weights)
            ]
        )
    
    def __mul__(self, scalar: float) -> 'DNNApprox[X]':
        """
        Multiply neural network weights by a scalar.
        
        Args:
            scalar: A scalar value
            
        Returns:
            New neural network with scaled weights
        """
        return replace(
            self,
            weights=[
                replace(w, weights=w.weights * scalar)
                for w in self.weights
            ]
        )
    
    def update_with_gradient(
        self,
        gradient: Gradient['DNNApprox[X]']
    ) -> 'DNNApprox[X]':
        """
        Update weights using gradient.
        
        Args:
            gradient: Gradient object
            
        Returns:
            Updated neural network
        """
        if not isinstance(gradient.function_approx, DNNApprox):
            raise TypeError("Expected DNNApprox gradient")
        
        return replace(
            self,
            weights=[
                w.update(g.weights)
                for w, g in zip(self.weights, gradient.function_approx.weights)
            ]
        )
    
    def solve(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: float = None
    ) -> 'DNNApprox[X]':
        """
        Fit the neural network to data.
        
        Args:
            xy_vals_seq: Sequence of (x, y) pairs
            error_tolerance: Optional convergence tolerance
            
        Returns:
            Fitted neural network
        """
        # Use default tolerance if not provided
        tol = 1e-6 if error_tolerance is None else error_tolerance
        
        # Define convergence check function
        def done(a: 'DNNApprox[X]', b: 'DNNApprox[X]', tol=tol) -> bool:
            return a.within(b, tol)
        
        # Convert input to list to allow multiple iterations
        xy_vals_list = list(xy_vals_seq)
        
        # Create iterator of updates
        updates = self.iterate_updates(
            itertools.repeat(xy_vals_list)
        )
        
        # Train until convergence
        return converged(updates, done_func=done)
    
    def within(self, other: 'DNNApprox[X]', tolerance: float) -> bool:
        """
        Check if network is within tolerance of another network.
        
        Args:
            other: Another neural network of the same architecture
            tolerance: Tolerance for comparison
            
        Returns:
            True if all weights are within tolerance, False otherwise
        """
        if not isinstance(other, DNNApprox):
            return False
        
        return all(
            w1.within(w2, tolerance)
            for w1, w2 in zip(self.weights, other.weights)
        )
    
    def iterate_updates(
        self,
        xy_seq_stream: Iterable[Iterable[Tuple[X, float]]]
    ) -> Iterable['DNNApprox[X]']:
        """
        Perform a series of updates with different data batches.
        
        Args:
            xy_seq_stream: Iterator of (x, y) pair sequences
            
        Returns:
            Iterator of updated networks
        """
        return accumulate(
            xy_seq_stream,
            lambda fa, xy: fa.update(xy),
            initial=self
        )
