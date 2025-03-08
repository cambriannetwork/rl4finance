"""
Standalone implementation of the Asset Allocation Discrete model.
This script contains a cleaned-up version of the code with more common Python syntax.
"""

import numpy as np
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, TypeVar, Iterable
from operator import itemgetter
import itertools
from abc import ABC, abstractmethod

# Type variables for generic functions and classes
A = TypeVar('A')  # Action type
S = TypeVar('S')  # State type
X = TypeVar('X')  # Input type for function approximation
F = TypeVar('F', bound='FunctionApprox')  # Function approximation type

# Small number to avoid division by zero
SMALL_NUM = 1e-6

###################
# Distribution    #
###################

class Distribution:
    """Base class for probability distributions that can be sampled."""
    
    def sample(self):
        """Return a random sample from this distribution."""
        raise NotImplementedError
    
    def sample_n(self, n: int):
        """Return n samples from this distribution."""
        return [self.sample() for _ in range(n)]
    
    def expectation(self, f: Callable):
        """Return the expectation of f(X) where X is the random variable."""
        raise NotImplementedError


class SampledDistribution(Distribution):
    """A distribution defined by a sampling function."""
    
    def __init__(self, sampler: Callable, expectation_samples: int = 1000):
        """
        Args:
            sampler: Function that returns a sample when called
            expectation_samples: Number of samples to use for expectation approximation
        """
        self.sampler = sampler
        self.expectation_samples = expectation_samples
    
    def sample(self):
        return self.sampler()
    
    def expectation(self, f: Callable):
        """Approximate expectation by averaging over samples."""
        return sum(f(self.sample()) for _ in range(self.expectation_samples)) / self.expectation_samples


class Gaussian(SampledDistribution):
    """Gaussian (normal) distribution with given mean and standard deviation."""
    
    def __init__(self, μ: float, σ: float, expectation_samples: int = 1000):
        """
        Args:
            μ: Mean of the distribution
            σ: Standard deviation of the distribution
            expectation_samples: Number of samples for expectation approximation
        """
        self.μ = μ
        self.σ = σ
        super().__init__(
            sampler=lambda: np.random.normal(loc=self.μ, scale=self.σ),
            expectation_samples=expectation_samples
        )


class Choose(Distribution):
    """Uniform distribution over a finite set of options."""
    
    def __init__(self, options: Iterable):
        """
        Args:
            options: Collection of items to choose from with equal probability
        """
        self.options = list(options)
    
    def sample(self):
        import random
        return random.choice(self.options)
    
    def expectation(self, f: Callable):
        """Calculate expectation by averaging over all possible outcomes."""
        return sum(f(option) for option in self.options) / len(self.options)


@dataclass(frozen=True)
class Constant(Distribution):
    """A distribution with a single outcome that has probability 1."""
    value: any
    
    def sample(self):
        return self.value
    
    def expectation(self, f: Callable):
        return f(self.value)


###################
# State Classes   #
###################

class State:
    """Base class for states in a Markov process."""
    
    def __init__(self, state_value):
        self.state = state_value
    
    def on_non_terminal(self, f: Callable, default):
        """Apply function f if this is a non-terminal state, otherwise return default."""
        if isinstance(self, NonTerminal):
            return f(self)
        else:
            return default


class Terminal(State):
    """Terminal state in a Markov process."""
    
    def __init__(self, state_value):
        super().__init__(state_value)


class NonTerminal(State):
    """Non-terminal state in a Markov process."""
    
    def __init__(self, state_value):
        super().__init__(state_value)
    
    def __eq__(self, other):
        return self.state == other.state
    
    def __lt__(self, other):
        return self.state < other.state


###################
# MDP            #
###################

class MarkovDecisionProcess:
    """Base class for Markov Decision Processes."""
    
    def actions(self, state: NonTerminal):
        """Return the available actions in the given state."""
        raise NotImplementedError
    
    def step(self, state: NonTerminal, action):
        """
        Take an action in a state and return a distribution over
        (next_state, reward) pairs.
        """
        raise NotImplementedError


###################
# Policy         #
###################

class Policy:
    """A policy maps states to distributions over actions."""
    
    def act(self, state: NonTerminal):
        """Return a distribution over actions for the given state."""
        raise NotImplementedError


@dataclass(frozen=True)
class DeterministicPolicy(Policy):
    """A policy that returns a single action for each state."""
    action_for: Callable
    
    def act(self, state: NonTerminal):
        return Constant(self.action_for(state.state))


###################
# Utilities      #
###################

def iterate(step_func: Callable, start_value):
    """
    Generate a sequence by repeatedly applying a function to its own result.
    
    Args:
        step_func: Function to apply repeatedly
        start_value: Initial value
        
    Returns:
        Iterator yielding: start_value, f(start_value), f(f(start_value)), ...
    """
    state = start_value
    while True:
        yield state
        state = step_func(state)


def converge(values_iterator, done_func: Callable):
    """
    Read from an iterator until two consecutive values satisfy the done function.
    
    Args:
        values_iterator: Iterator of values
        done_func: Function that takes two consecutive values and returns True if converged
        
    Returns:
        Iterator that stops when convergence is detected
    """
    a = next(values_iterator, None)
    if a is None:
        return
    
    yield a
    
    for b in values_iterator:
        yield b
        if done_func(a, b):
            return
        a = b


def converged(values_iterator, done_func: Callable = None, *, done = None):
    """
    Return the final value when an iterator converges according to done_func.
    
    Args:
        values_iterator: Iterator of values
        done_func: Function that takes two consecutive values and returns True if converged
        done: Alternative name for done_func (for backward compatibility)
        
    Returns:
        The final value after convergence
    """
    # Handle both done_func and done parameters for compatibility
    if done is not None:
        done_func = done
    
    if done_func is None:
        raise ValueError("Either done_func or done must be provided")
    
    result = None
    for x in converge(values_iterator, done_func):
        result = x
    
    if result is None:
        raise ValueError("converged called on an empty iterator")
    
    return result


def accumulate(iterable, func, *, initial=None):
    """
    Make an iterator that returns accumulated results of a binary function.
    
    Similar to itertools.accumulate but with an initial value.
    
    Args:
        iterable: Input iterable
        func: Binary function to apply
        initial: Optional initial value
        
    Returns:
        Iterator of accumulated values
    """
    if initial is not None:
        iterable = itertools.chain([initial], iterable)
    
    return itertools.accumulate(iterable, func)


###################
# Function Approx #
###################

class FunctionApprox(ABC):
    """
    Interface for function approximations.
    
    Approximates a function X → ℝ that can be evaluated at specific points
    and updated with additional (X, ℝ) points.
    """
    
    @abstractmethod
    def __add__(self, other):
        pass
    
    @abstractmethod
    def __mul__(self, scalar: float):
        pass
    
    @abstractmethod
    def objective_gradient(self, xy_vals_seq, obj_deriv_out_fun: Callable):
        """
        Compute the gradient of an objective function with respect to parameters.
        
        Args:
            xy_vals_seq: Sequence of (x, y) pairs
            obj_deriv_out_fun: Function to compute derivative of objective
            
        Returns:
            Gradient object
        """
        pass
    
    @abstractmethod
    def evaluate(self, x_values_seq):
        """
        Compute expected value of y for each x in x_values_seq.
        
        Args:
            x_values_seq: Sequence of x values
            
        Returns:
            Array of predicted y values
        """
        pass
    
    def __call__(self, x_value):
        """Evaluate the function at a single point."""
        return self.evaluate([x_value]).item()
    
    @abstractmethod
    def update_with_gradient(self, gradient):
        """
        Update parameters using the given gradient.
        
        Args:
            gradient: Gradient object
            
        Returns:
            Updated FunctionApprox
        """
        pass
    
    def update(self, xy_vals_seq):
        """
        Update parameters based on (x, y) pairs.
        
        Args:
            xy_vals_seq: Sequence of (x, y) pairs
            
        Returns:
            Updated FunctionApprox
        """
        def deriv_func(x, y):
            return self.evaluate(x) - np.array(y)
        
        return self.update_with_gradient(
            self.objective_gradient(xy_vals_seq, deriv_func)
        )
    
    @abstractmethod
    def solve(self, xy_vals_seq, error_tolerance=None):
        """
        Fit parameters to the given data.
        
        Args:
            xy_vals_seq: Sequence of (x, y) pairs
            error_tolerance: Optional error tolerance for convergence
            
        Returns:
            Fitted FunctionApprox
        """
        pass
    
    @abstractmethod
    def within(self, other, tolerance: float) -> bool:
        """
        Check if this function approximation is within tolerance of another.
        
        Args:
            other: Another FunctionApprox
            tolerance: Tolerance for comparison
            
        Returns:
            True if within tolerance, False otherwise
        """
        pass


@dataclass(frozen=True)
class Gradient:
    """Gradient of a FunctionApprox with respect to its parameters."""
    function_approx: FunctionApprox
    
    def __add__(self, other):
        return Gradient(self.function_approx + other.function_approx)
    
    def __mul__(self, scalar: float):
        return Gradient(self.function_approx * scalar)


@dataclass(frozen=True)
class AdamGradient:
    """Parameters for Adam gradient descent optimizer."""
    learning_rate: float
    decay1: float
    decay2: float
    
    @staticmethod
    def default_settings():
        return AdamGradient(
            learning_rate=0.001,
            decay1=0.9,
            decay2=0.999
        )


@dataclass(frozen=True)
class Weights:
    """Weights for a neural network with Adam optimizer state."""
    adam_gradient: AdamGradient
    time: int
    weights: np.ndarray
    adam_cache1: np.ndarray
    adam_cache2: np.ndarray
    
    @staticmethod
    def create(weights, adam_gradient=None, adam_cache1=None, adam_cache2=None):
        """Create a new Weights object with optional Adam optimizer state."""
        if adam_gradient is None:
            adam_gradient = AdamGradient.default_settings()
        
        return Weights(
            adam_gradient=adam_gradient,
            time=0,
            weights=weights,
            adam_cache1=np.zeros_like(weights) if adam_cache1 is None else adam_cache1,
            adam_cache2=np.zeros_like(weights) if adam_cache2 is None else adam_cache2
        )
    
    def update(self, gradient: np.ndarray):
        """Update weights using Adam optimizer."""
        time = self.time + 1
        new_adam_cache1 = self.adam_gradient.decay1 * self.adam_cache1 + (1 - self.adam_gradient.decay1) * gradient
        new_adam_cache2 = self.adam_gradient.decay2 * self.adam_cache2 + (1 - self.adam_gradient.decay2) * gradient ** 2
        
        # Bias correction
        corrected_m = new_adam_cache1 / (1 - self.adam_gradient.decay1 ** time)
        corrected_v = new_adam_cache2 / (1 - self.adam_gradient.decay2 ** time)
        
        # Update weights
        new_weights = self.weights - self.adam_gradient.learning_rate * corrected_m / (np.sqrt(corrected_v) + SMALL_NUM)
        
        return replace(
            self,
            time=time,
            weights=new_weights,
            adam_cache1=new_adam_cache1,
            adam_cache2=new_adam_cache2,
        )
    
    def within(self, other, tolerance: float) -> bool:
        """Check if weights are within tolerance of another set of weights."""
        return np.all(np.abs(self.weights - other.weights) <= tolerance).item()


@dataclass(frozen=True)
class DNNSpec:
    """Specification for a neural network architecture."""
    neurons: Sequence[int]  # Number of neurons in each hidden layer
    bias: bool  # Whether to include bias terms
    hidden_activation: Callable  # Activation function for hidden layers
    hidden_activation_deriv: Callable  # Derivative of hidden activation
    output_activation: Callable  # Activation function for output layer
    output_activation_deriv: Callable  # Derivative of output activation


@dataclass(frozen=True)
class DNNApprox(FunctionApprox):
    """Neural network function approximation."""
    
    feature_functions: Sequence[Callable]  # Functions to extract features from inputs
    dnn_spec: DNNSpec  # Neural network architecture
    regularization_coeff: float  # L2 regularization coefficient
    weights: Sequence[Weights]  # Weights for each layer
    
    @staticmethod
    def create(
        feature_functions: Sequence[Callable],
        dnn_spec: DNNSpec,
        adam_gradient=None,
        regularization_coeff=0.0,
        weights=None
    ):
        """
        Create a new neural network function approximation.
        
        Args:
            feature_functions: Functions to extract features from inputs
            dnn_spec: Neural network architecture
            adam_gradient: Optional Adam optimizer settings
            regularization_coeff: L2 regularization coefficient
            weights: Optional pre-initialized weights
            
        Returns:
            New DNNApprox instance
        """
        if adam_gradient is None:
            adam_gradient = AdamGradient.default_settings()
            
        if weights is None:
            # Initialize weights for each layer
            inputs = [len(feature_functions)] + [
                n + (1 if dnn_spec.bias else 0)
                for i, n in enumerate(dnn_spec.neurons)
            ]
            outputs = list(dnn_spec.neurons) + [1]
            
            # Initialize with Xavier/Glorot initialization
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
    
    def get_feature_values(self, x_values_seq):
        """Extract features from input values."""
        return np.array(
            [[f(x) for f in self.feature_functions] for x in x_values_seq]
        )
    
    def forward_propagation(self, x_values_seq):
        """
        Perform forward pass through the neural network.
        
        Args:
            x_values_seq: Sequence of input values
            
        Returns:
            List of activations for each layer, with the final output as the last element
        """
        inp = self.get_feature_values(x_values_seq)
        activations = [inp]
        
        # Process hidden layers
        for w in self.weights[:-1]:
            out = self.dnn_spec.hidden_activation(np.dot(inp, w.weights.T))
            
            # Add bias term if specified
            if self.dnn_spec.bias:
                inp = np.insert(out, 0, 1., axis=1)
            else:
                inp = out
                
            activations.append(inp)
        
        # Process output layer
        output = self.dnn_spec.output_activation(np.dot(inp, self.weights[-1].weights.T))[:, 0]
        activations.append(output)
        
        return activations
    
    def evaluate(self, x_values_seq):
        """Evaluate the network on input values."""
        return self.forward_propagation(x_values_seq)[-1]
    
    def backward_propagation(self, fwd_prop, obj_deriv_out):
        """
        Perform backpropagation to compute gradients.
        
        Args:
            fwd_prop: Result of forward propagation (activations)
            obj_deriv_out: Derivative of objective function with respect to output
            
        Returns:
            List of gradients for each layer's weights
        """
        deriv = obj_deriv_out.reshape(1, -1)
        
        # Check if fwd_prop has enough elements
        if len(fwd_prop) < 2:
            # Handle the case where there are no hidden layers
            gradients = [np.dot(deriv, fwd_prop[0]) / deriv.shape[1]]
        else:
            gradients = [np.dot(deriv, fwd_prop[-1]) / deriv.shape[1]]
            
            # Backpropagate through hidden layers
            for i in reversed(range(len(self.weights) - 1)):
                # Ensure i+1 is within bounds
                layer_idx = min(i + 1, len(fwd_prop) - 1)
                
                # Compute derivative with respect to layer output
                deriv = np.dot(self.weights[i + 1].weights.T, deriv) * \
                    self.dnn_spec.hidden_activation_deriv(fwd_prop[layer_idx].T)
                
                # Remove bias term gradient if present
                if self.dnn_spec.bias:
                    deriv = deriv[1:]
                    
                # Compute gradient for layer weights (ensure i is within bounds)
                input_idx = min(i, len(fwd_prop) - 1)
                gradients.append(np.dot(deriv, fwd_prop[input_idx]) / deriv.shape[1])
        
        return gradients[::-1]  # Reverse to match layer order
    
    def objective_gradient(self, xy_vals_seq, obj_deriv_out_fun):
        """
        Compute gradient of objective function with respect to network parameters.
        
        Args:
            xy_vals_seq: Sequence of (x, y) pairs
            obj_deriv_out_fun: Function to compute derivative of objective
            
        Returns:
            Gradient object
        """
        x_vals, y_vals = zip(*xy_vals_seq)
        obj_deriv_out = obj_deriv_out_fun(x_vals, y_vals)
        
        # Forward and backward passes
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
    
    def __add__(self, other):
        """Add two neural networks by adding their weights."""
        if not isinstance(other, DNNApprox):
            raise TypeError("Can only add DNNApprox instances")
            
        return replace(
            self,
            weights=[
                replace(w, weights=w.weights + o.weights)
                for w, o in zip(self.weights, other.weights)
            ]
        )
    
    def __mul__(self, scalar: float):
        """Multiply neural network weights by a scalar."""
        return replace(
            self,
            weights=[
                replace(w, weights=w.weights * scalar)
                for w in self.weights
            ]
        )
    
    def update_with_gradient(self, gradient):
        """Update weights using gradient."""
        if not isinstance(gradient.function_approx, DNNApprox):
            raise TypeError("Expected DNNApprox gradient")
            
        return replace(
            self,
            weights=[
                w.update(g.weights)
                for w, g in zip(self.weights, gradient.function_approx.weights)
            ]
        )
    
    def solve(self, xy_vals_seq, error_tolerance=None):
        """
        Fit the neural network to data.
        
        Args:
            xy_vals_seq: Sequence of (x, y) pairs
            error_tolerance: Optional convergence tolerance
            
        Returns:
            Fitted neural network
        """
        tol = 1e-6 if error_tolerance is None else error_tolerance
        
        def done(a, b, tol=tol):
            return a.within(b, tol)
        
        # Train until convergence
        return converged(
            self.iterate_updates(itertools.repeat(list(xy_vals_seq))),
            done=done
        )
    
    def within(self, other, tolerance: float) -> bool:
        """Check if network is within tolerance of another network."""
        if not isinstance(other, DNNApprox):
            return False
            
        return all(
            np.all(np.abs(w1.weights - w2.weights) <= tolerance)
            for w1, w2 in zip(self.weights, other.weights)
        )
    
    def iterate_updates(self, xy_seq_stream):
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


###################
# ADP Functions   #
###################

# Type aliases for clarity
ValueFunctionApprox = FunctionApprox  # For approximating V(s)
QValueFunctionApprox = FunctionApprox  # For approximating Q(s,a)


def extended_vf(vf, s):
    """Evaluate value function, returning 0 for terminal states."""
    if isinstance(s, NonTerminal):
        return vf(s)
    else:
        return 0.0


def back_opt_vf_and_policy(mdp_f0_mu_triples, γ, num_state_samples, error_tolerance):
    """
    Use backwards induction to find optimal value function and policy.
    
    Args:
        mdp_f0_mu_triples: Sequence of (MDP, initial function approx, state distribution)
        γ: Discount factor
        num_state_samples: Number of states to sample
        error_tolerance: Convergence tolerance
        
    Returns:
        Iterator of (value function, policy) pairs
    """
    vp = []
    
    for i, (mdp, approx0, mu) in enumerate(reversed(mdp_f0_mu_triples)):
        # Define return function for this time step
        def return_(s_r, i=i):
            s1, r = s_r
            return r + γ * (extended_vf(vp[i-1][0], s1) if i > 0 else 0.)
        
        # Solve for value function
        this_v = approx0.solve(
            [(s, max(mdp.step(s, a).expectation(return_)
                     for a in mdp.actions(s)))
             for s in mu.sample_n(num_state_samples)],
            error_tolerance
        )
        
        # Define deterministic policy
        def deter_policy(state):
            return max(
                ((mdp.step(NonTerminal(state), a).expectation(return_), a)
                 for a in mdp.actions(NonTerminal(state))),
                key=itemgetter(0)
            )[1]
        
        vp.append((this_v, DeterministicPolicy(deter_policy)))
    
    return reversed(vp)


def back_opt_qvf(mdp_f0_mu_triples, γ, num_state_samples, error_tolerance):
    """
    Use backwards induction to find optimal Q-value function.
    
    Args:
        mdp_f0_mu_triples: Sequence of (MDP, initial Q-function approx, state distribution)
        γ: Discount factor
        num_state_samples: Number of states to sample
        error_tolerance: Convergence tolerance
        
    Returns:
        Iterator of Q-value functions
    """
    horizon = len(mdp_f0_mu_triples)
    qvf = []
    
    for i, (mdp, approx0, mu) in enumerate(reversed(mdp_f0_mu_triples)):
        # Define return function for this time step
        def return_(s_r, i=i):
            s1, r = s_r
            
            # Calculate max Q-value for next state if not terminal
            next_return = 0.0
            if i > 0 and isinstance(s1, NonTerminal):
                next_mdp = mdp_f0_mu_triples[horizon - i][0]
                next_return = max(
                    qvf[i-1]((s1, a)) for a in next_mdp.actions(s1)
                )
                
            return r + γ * next_return
        
        # Solve for Q-value function
        this_qvf = approx0.solve(
            [((s, a), mdp.step(s, a).expectation(return_))
             for s in mu.sample_n(num_state_samples) for a in mdp.actions(s)],
            error_tolerance
        )
        
        qvf.append(this_qvf)
    
    return reversed(qvf)


###################
# Asset Allocation#
###################

@dataclass(frozen=True)
class AssetAllocDiscrete:
    """
    Asset allocation model with discrete action choices.
    
    This model determines optimal allocation between risky and riskless assets
    over multiple time steps.
    """
    risky_return_distributions: Sequence[Distribution]  # Distribution of risky asset returns
    riskless_returns: Sequence[float]  # Risk-free returns
    utility_func: Callable[[float], float]  # Utility function
    risky_alloc_choices: Sequence[float]  # Possible allocations to risky asset
    feature_functions: Sequence[Callable]  # Feature functions for function approximation
    dnn_spec: DNNSpec  # Neural network specification
    initial_wealth_distribution: Distribution  # Distribution of initial wealth
    
    def time_steps(self) -> int:
        """Return the number of time steps in the model."""
        return len(self.risky_return_distributions)
    
    def uniform_actions(self) -> Choose:
        """Return a uniform distribution over allocation choices."""
        return Choose(self.risky_alloc_choices)
    
    def get_mdp(self, t: int) -> MarkovDecisionProcess:
        """
        Create an MDP for time step t.
        
        State is wealth W_t, action is investment in risky asset (x_t).
        Investment in riskless asset is W_t - x_t.
        
        Args:
            t: Time step
            
        Returns:
            MarkovDecisionProcess for time step t
        """
        distr = self.risky_return_distributions[t]
        rate = self.riskless_returns[t]
        alloc_choices = self.risky_alloc_choices
        steps = self.time_steps()
        utility_f = self.utility_func
        
        class AssetAllocMDP(MarkovDecisionProcess):
            def step(self, wealth: NonTerminal, alloc: float) -> SampledDistribution:
                """
                Take allocation action from current wealth state.
                
                Args:
                    wealth: Current wealth state
                    alloc: Amount to allocate to risky asset
                    
                Returns:
                    Distribution over (next_state, reward) pairs
                """
                def sr_sampler_func():
                    # Calculate next wealth based on returns
                    next_wealth = alloc * (1 + distr.sample()) + (wealth.state - alloc) * (1 + rate)
                    
                    # Reward is utility at final time step, 0 otherwise
                    reward = utility_f(next_wealth) if t == steps - 1 else 0.0
                    
                    # Next state is terminal at final time step
                    next_state = Terminal(next_wealth) if t == steps - 1 else NonTerminal(next_wealth)
                    
                    return (next_state, reward)
                
                return SampledDistribution(sampler=sr_sampler_func, expectation_samples=1000)
            
            def actions(self, wealth: NonTerminal) -> Sequence[float]:
                """Return available actions (allocation choices)."""
                return alloc_choices
        
        return AssetAllocMDP()
    
    def get_qvf_func_approx(self) -> DNNApprox:
        """
        Create a neural network for Q-value function approximation.
        
        Returns:
            DNNApprox for Q-value function
        """
        adam_gradient = AdamGradient(
            learning_rate=0.1,
            decay1=0.9,
            decay2=0.999
        )
        
        # Create feature functions that work with (state, action) pairs
        ffs = []
        for f in self.feature_functions:
            def this_f(pair, f=f):
                return f((pair[0].state, pair[1]))
            ffs.append(this_f)
        
        return DNNApprox.create(
            feature_functions=ffs,
            dnn_spec=self.dnn_spec,
            adam_gradient=adam_gradient
        )
    
    def get_states_distribution(self, t: int) -> SampledDistribution:
        """
        Create a distribution over states at time t.
        
        Args:
            t: Time step
            
        Returns:
            Distribution over non-terminal states at time t
        """
        actions_distr = self.uniform_actions()
        
        def states_sampler_func():
            # Start with initial wealth
            wealth = self.initial_wealth_distribution.sample()
            
            # Simulate forward to time t
            for i in range(t):
                distr = self.risky_return_distributions[i]
                rate = self.riskless_returns[i]
                alloc = actions_distr.sample()
                wealth = alloc * (1 + distr.sample()) + (wealth - alloc) * (1 + rate)
            
            return NonTerminal(wealth)
        
        return SampledDistribution(states_sampler_func)
    
    def backward_induction_qvf(self) -> Iterator:
        """
        Use backward induction to find optimal Q-value functions.
        
        Returns:
            Iterator of Q-value functions for each time step
        """
        # Initialize function approximation
        init_fa = self.get_qvf_func_approx()
        
        # Create MDP, function approximation, and state distribution for each time step
        mdp_f0_mu_triples = [
            (
                self.get_mdp(i),
                init_fa,
                self.get_states_distribution(i)
            ) 
            for i in range(self.time_steps())
        ]
        
        # Parameters for backward induction
        num_state_samples = 300
        error_tolerance = 1e-6
        
        # Perform backward induction
        return back_opt_qvf(
            mdp_f0_mu_triples=mdp_f0_mu_triples,
            γ=1.0,  # No discounting
            num_state_samples=num_state_samples,
            error_tolerance=error_tolerance
        )
    
    def get_vf_func_approx(self, ff: Sequence[Callable]) -> DNNApprox:
        """
        Create a neural network for value function approximation.
        
        Args:
            ff: Feature functions for states
            
        Returns:
            DNNApprox for value function
        """
        adam_gradient = AdamGradient(
            learning_rate=0.1,
            decay1=0.9,
            decay2=0.999
        )
        
        return DNNApprox.create(
            feature_functions=ff,
            dnn_spec=self.dnn_spec,
            adam_gradient=adam_gradient
        )
    
    def backward_induction_vf_and_pi(self, ff: Sequence[Callable]) -> Iterator:
        """
        Use backward induction to find optimal value functions and policies.
        
        Args:
            ff: Feature functions for states
            
        Returns:
            Iterator of (value function, policy) pairs for each time step
        """
        # Initialize function approximation
        init_fa = self.get_vf_func_approx(ff)
        
        # Create MDP, function approximation, and state distribution for each time step
        mdp_f0_mu_triples = [
            (
                self.get_mdp(i),
                init_fa,
                self.get_states_distribution(i)
            ) 
            for i in range(self.time_steps())
        ]
        
        # Parameters for backward induction
        num_state_samples = 300
        error_tolerance = 1e-8
        
        # Perform backward induction
        return back_opt_vf_and_policy(
            mdp_f0_mu_triples=mdp_f0_mu_triples,
            γ=1.0,  # No discounting
            num_state_samples=num_state_samples,
            error_tolerance=error_tolerance
        )


if __name__ == '__main__':
    from pprint import pprint
    
    # Model parameters
    steps = 4
    μ = 0.13  # Mean return of risky asset
    σ = 0.2   # Standard deviation of risky asset return
    r = 0.07  # Risk-free rate
    a = 1.0   # Risk aversion parameter
    init_wealth = 1.0
    init_wealth_stdev = 0.1
    
    # Calculate base allocation using analytical formula
    excess = μ - r
    var = σ * σ
    base_alloc = excess / (a * var)
    
    # Create distributions and utility function
    risky_ret = [Gaussian(μ=μ, σ=σ) for _ in range(steps)]
    riskless_ret = [r for _ in range(steps)]
    utility_function = lambda x: -np.exp(-a * x) / a
    
    # Define allocation choices around the base allocation
    alloc_choices = np.linspace(
        2/3 * base_alloc,
        4/3 * base_alloc,
        11
    )
    
    # Define feature functions for function approximation
    feature_funcs = [
        lambda _: 1.,  # Bias term
        lambda w_x: w_x[0],  # Wealth
        lambda w_x: w_x[1],  # Allocation
        lambda w_x: w_x[1] * w_x[1]  # Allocation squared
    ]
    
    # Define neural network architecture (linear in this case)
    dnn = DNNSpec(
        neurons=[],  # No hidden layers
        bias=False,
        hidden_activation=lambda x: x,  # Identity
        hidden_activation_deriv=lambda y: np.ones_like(y),
        output_activation=lambda x: -np.sign(a) * np.exp(-x),
        output_activation_deriv=lambda y: -y
    )
    
    # Create initial wealth distribution
    init_wealth_distr = Gaussian(μ=init_wealth, σ=init_wealth_stdev)
    
    # Create asset allocation model
    aad = AssetAllocDiscrete(
        risky_return_distributions=risky_ret,
        riskless_returns=riskless_ret,
        utility_func=utility_function,
        risky_alloc_choices=alloc_choices,
        feature_functions=feature_funcs,
        dnn_spec=dnn,
        initial_wealth_distribution=init_wealth_distr
    )
    
    # Uncomment to run value function and policy backward induction
    # vf_ff = [lambda _: 1., lambda w: w.state]
    # it_vf = aad.backward_induction_vf_and_pi(vf_ff)
    # 
    # print("Backward Induction: VF And Policy")
    # print("---------------------------------")
    # print()
    # for t, (v, p) in enumerate(it_vf):
    #     print(f"Time {t:d}")
    #     print()
    #     opt_alloc = p.action_for(init_wealth)
    #     val = v(NonTerminal(init_wealth))
    #     print(f"Opt Risky Allocation = {opt_alloc:.2f}, Opt Val = {val:.3f}")
    #     print("Weights")
    #     for w in v.weights:
    #         print(w.weights)
    #     print()
    
    # Run Q-value function backward induction
    it_qvf = aad.backward_induction_qvf()
    
    print("Backward Induction on Q-Value Function")
    print("--------------------------------------")
    print()
    for t, q in enumerate(it_qvf):
        print(f"Time {t:d}")
        print()
        # Find optimal allocation and value
        opt_alloc = max(
            ((q((NonTerminal(init_wealth), ac)), ac) for ac in alloc_choices),
            key=itemgetter(0)
        )[1]
        val = max(q((NonTerminal(init_wealth), ac)) for ac in alloc_choices)
        print(f"Opt Risky Allocation = {opt_alloc:.3f}, Opt Val = {val:.3f}")
        print("Optimal Weights below:")
        for wts in q.weights:
            pprint(wts.weights)
        print()
    
    print("Analytical Solution")
    print("-------------------")
    print()
    
    # Calculate and print analytical solution for each time step
    for t in range(steps):
        print(f"Time {t:d}")
        print()
        left = steps - t
        growth = (1 + r) ** (left - 1)
        alloc = base_alloc / growth
        vval = -np.exp(-excess * excess * left / (2 * var) - a * growth * (1 + r) * init_wealth) / a
        
        # Analytical weights
        bias_wt = excess * excess * (left - 1) / (2 * var) + np.log(np.abs(a))
        w_t_wt = a * growth * (1 + r)
        x_t_wt = a * excess * growth
        x_t2_wt = -var * (a * growth) ** 2 / 2
        
        print(f"Opt Risky Allocation = {alloc:.3f}, Opt Val = {vval:.3f}")
        print(f"Bias Weight = {bias_wt:.3f}")
        print(f"W_t Weight = {w_t_wt:.3f}")
        print(f"x_t Weight = {x_t_wt:.3f}")
        print(f"x_t^2 Weight = {x_t2_wt:.3f}")
        print()
