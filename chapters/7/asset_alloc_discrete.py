"""
Standalone implementation of the Asset Allocation Discrete model.
This script contains all necessary code from the RL package to run the model.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import (Callable, Dict, Generic, Iterator, Iterable, List, 
                   Mapping, Optional, Sequence, Tuple, TypeVar)
from operator import itemgetter
import numpy as np
import random
import time
import argparse

# Type variables
S = TypeVar('S')
A = TypeVar('A')
X = TypeVar('X')
F = TypeVar('F', bound='FunctionApprox')

# Constants
SMALL_NUM = 1e-6

###################
# Distribution.py #
###################

class Distribution(ABC, Generic[A]):
    '''A probability distribution that we can sample.'''
    
    @abstractmethod
    def sample(self) -> A:
        '''Return a random sample from this distribution.'''
        pass

    def sample_n(self, n: int) -> Sequence[A]:
        '''Return n samples from this distribution.'''
        return [self.sample() for _ in range(n)]

    @abstractmethod
    def expectation(self, f: Callable[[A], float]) -> float:
        '''Return the expectation of f(X) where X is the
        random variable for the distribution and f is an
        arbitrary function from X to float
        '''
        pass


class SampledDistribution(Distribution[A]):
    '''A distribution defined by a function to sample it.'''
    
    sampler: Callable[[], A]
    expectation_samples: int

    def __init__(
        self,
        sampler: Callable[[], A],
        expectation_samples: int = 1000
    ):
        self.sampler = sampler
        self.expectation_samples = expectation_samples

    def sample(self) -> A:
        return self.sampler()

    def expectation(self, f: Callable[[A], float]) -> float:
        '''Return a sampled approximation of the expectation of f(X) for some f.'''
        # Get verbose level from globals if available
        verbose = globals().get('verbose', 0)
        
        # Only show progress for large sample sizes and only during initial setup
        # Not during the main backward induction calculations
        show_progress = verbose > 1 and self.expectation_samples > 100
        
        # Track if we're in the backward induction phase
        in_backward_induction = globals().get('_in_backward_induction', False)
        
        # Don't show progress during backward induction to avoid excessive output
        if in_backward_induction:
            show_progress = False
        
        if show_progress:
            print(f"    Monte Carlo sampling with {self.expectation_samples} samples...", end="\r")
            
        total = 0.0
        for i in range(self.expectation_samples):
            total += f(self.sample())
            
            # Show progress every 10% of samples
            if show_progress and i % max(1, self.expectation_samples // 10) == 0 and i > 0:
                progress = i / self.expectation_samples * 100
                print(f"    Monte Carlo progress: {progress:.1f}% complete", end="\r")
                
        if show_progress:
            print("    Monte Carlo sampling completed.                    ")
            
        return total / self.expectation_samples


class Gaussian(SampledDistribution[float]):
    '''A Gaussian distribution with the given μ and σ.'''

    μ: float
    σ: float

    def __init__(self, μ: float, σ: float, expectation_samples: int = None):
        self.μ = μ
        self.σ = σ
        # Use the global expectation_samples parameter if none is provided
        samples = expectation_samples if expectation_samples is not None else globals().get('expectation_samples', 1000)
        super().__init__(
            sampler=lambda: np.random.normal(loc=self.μ, scale=self.σ),
            expectation_samples=samples
        )


class Choose(Distribution[A]):
    '''Select an element of the given list uniformly at random.'''

    options: Sequence[A]

    def __init__(self, options: Iterable[A]):
        self.options = list(options)

    def sample(self) -> A:
        return random.choice(self.options)

    def expectation(self, f: Callable[[A], float]) -> float:
        '''Calculate expectation by averaging over all possible outcomes.'''
        return sum(f(option) for option in self.options) / len(self.options)


######################
# Markov Process.py #
######################

class State(ABC, Generic[S]):
    state: S

    def on_non_terminal(
        self,
        f: Callable[[NonTerminal[S]], X],
        default: X
    ) -> X:
        if isinstance(self, NonTerminal):
            return f(self)
        else:
            return default


@dataclass(frozen=True)
class Terminal(State[S]):
    state: S


@dataclass(frozen=True)
class NonTerminal(State[S]):
    state: S
        
    def __eq__(self, other):
        return self.state == other.state

    def __lt__(self, other):
        return self.state < other.state


###############################
# Markov Decision Process.py #
###############################

class MarkovDecisionProcess(ABC, Generic[S, A]):
    @abstractmethod
    def actions(self, state: NonTerminal[S]) -> Iterable[A]:
        pass

    @abstractmethod
    def step(
        self,
        state: NonTerminal[S],
        action: A
    ) -> Distribution[Tuple[State[S], float]]:
        pass


###############
# Policy.py #
###############

class Policy(ABC, Generic[S, A]):
    '''A policy is a function that specifies what we should do (the
    action) at a given state of our MDP.
    '''
    @abstractmethod
    def act(self, state: NonTerminal[S]) -> Distribution[A]:
        '''A distribution of actions to take from the given non-terminal
        state.
        '''


@dataclass(frozen=True)
class DeterministicPolicy(Policy[S, A]):
    action_for: Callable[[S], A]

    def act(self, state: NonTerminal[S]) -> Distribution[A]:
        # We'll create a simple distribution that always returns the same action
        class ConstantDistribution(Distribution[A]):
            def __init__(self, value: A):
                self.value = value
                
            def sample(self) -> A:
                return self.value
                
            def expectation(self, f: Callable[[A], float]) -> float:
                return f(self.value)
                
        return ConstantDistribution(self.action_for(state.state))


#######################
# Function Approx.py #
#######################

class FunctionApprox(ABC, Generic[X]):
    '''Interface for function approximations.
    An object of this class approximates some function X ↦ ℝ in a way
    that can be evaluated at specific points in X and updated with
    additional (X, ℝ) points.
    '''

    @abstractmethod
    def __add__(self: F, other: F) -> F:
        pass

    @abstractmethod
    def __mul__(self: F, scalar: float) -> F:
        pass

    @abstractmethod
    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        '''Computes expected value of y for each x in
        x_values_seq (with the probability distribution
        function of y|x estimated as FunctionApprox)
        '''

    def __call__(self, x_value: X) -> float:
        return self.evaluate([x_value]).item()

    @abstractmethod
    def update(
        self: F,
        xy_vals_seq: Iterable[Tuple[X, float]]
    ) -> F:
        '''Update the internal parameters of the FunctionApprox
        based on incremental data provided in the form of (x,y)
        pairs as a xy_vals_seq data structure
        '''

    @abstractmethod
    def solve(
        self: F,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: Optional[float] = None
    ) -> F:
        '''Assuming the entire data set of (x,y) pairs is available
        in the form of the given input xy_vals_seq data structure,
        solve for the internal parameters of the FunctionApprox
        such that the internal parameters are fitted to xy_vals_seq.
        '''


@dataclass(frozen=True)
class Gradient(Generic[F]):
    function_approx: F

    def __add__(self, x: Gradient[F]) -> Gradient[F]:
        return Gradient(self.function_approx + x.function_approx)

    def __mul__(self: Gradient[F], x: float) -> Gradient[F]:
        return Gradient(self.function_approx * x)


@dataclass(frozen=True)
class AdamGradient:
    learning_rate: float
    decay1: float
    decay2: float

    @staticmethod
    def default_settings() -> AdamGradient:
        return AdamGradient(
            learning_rate=0.001,
            decay1=0.9,
            decay2=0.999
        )


@dataclass(frozen=True)
class Weights:
    adam_gradient: AdamGradient
    time: int
    weights: np.ndarray
    adam_cache1: np.ndarray
    adam_cache2: np.ndarray

    @staticmethod
    def create(
        weights: np.ndarray,
        adam_gradient: AdamGradient = AdamGradient.default_settings(),
        adam_cache1: Optional[np.ndarray] = None,
        adam_cache2: Optional[np.ndarray] = None
    ) -> Weights:
        return Weights(
            adam_gradient=adam_gradient,
            time=0,
            weights=weights,
            adam_cache1=np.zeros_like(
                weights
            ) if adam_cache1 is None else adam_cache1,
            adam_cache2=np.zeros_like(
                weights
            ) if adam_cache2 is None else adam_cache2
        )

    def update(self, gradient: np.ndarray) -> Weights:
        time: int = self.time + 1
        new_adam_cache1: np.ndarray = self.adam_gradient.decay1 * \
            self.adam_cache1 + (1 - self.adam_gradient.decay1) * gradient
        new_adam_cache2: np.ndarray = self.adam_gradient.decay2 * \
            self.adam_cache2 + (1 - self.adam_gradient.decay2) * gradient ** 2
        corrected_m: np.ndarray = new_adam_cache1 / \
            (1 - self.adam_gradient.decay1 ** time)
        corrected_v: np.ndarray = new_adam_cache2 / \
            (1 - self.adam_gradient.decay2 ** time)

        new_weights: np.ndarray = self.weights - \
            self.adam_gradient.learning_rate * corrected_m / \
            (np.sqrt(corrected_v) + SMALL_NUM)

        return replace(
            self,
            time=time,
            weights=new_weights,
            adam_cache1=new_adam_cache1,
            adam_cache2=new_adam_cache2,
        )


@dataclass(frozen=True)
class DNNSpec:
    neurons: Sequence[int]
    bias: bool
    hidden_activation: Callable[[np.ndarray], np.ndarray]
    hidden_activation_deriv: Callable[[np.ndarray], np.ndarray]
    output_activation: Callable[[np.ndarray], np.ndarray]
    output_activation_deriv: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class DNNApprox(FunctionApprox[X]):
    feature_functions: Sequence[Callable[[X], float]]
    dnn_spec: DNNSpec
    regularization_coeff: float
    weights: Sequence[Weights]

    @staticmethod
    def create(
        feature_functions: Sequence[Callable[[X], float]],
        dnn_spec: DNNSpec,
        adam_gradient: AdamGradient = AdamGradient.default_settings(),
        regularization_coeff: float = 0.,
        weights: Optional[Sequence[Weights]] = None
    ) -> DNNApprox[X]:
        if weights is None:
            inputs: Sequence[int] = [len(feature_functions)] + \
                [n + (1 if dnn_spec.bias else 0)
                 for i, n in enumerate(dnn_spec.neurons)]
            outputs: Sequence[int] = list(dnn_spec.neurons) + [1]
            wts = [Weights.create(
                weights=np.random.randn(output, inp) / np.sqrt(inp),
                adam_gradient=adam_gradient
            ) for inp, output in zip(inputs, outputs)]
        else:
            wts = weights

        return DNNApprox(
            feature_functions=feature_functions,
            dnn_spec=dnn_spec,
            regularization_coeff=regularization_coeff,
            weights=wts
        )

    def get_feature_values(self, x_values_seq: Iterable[X]) -> np.ndarray:
        return np.array(
            [[f(x) for f in self.feature_functions] for x in x_values_seq]
        )

    def forward_propagation(
        self,
        x_values_seq: Iterable[X]
    ) -> Sequence[np.ndarray]:
        """
        :param x_values_seq: a n-length iterable of input points
        :return: list of length (L+2) where the first (L+1) values
                 each represent the 2-D input arrays (of size n x |i_l|),
                 for each of the (L+1) layers (L of which are hidden layers),
                 and the last value represents the output of the DNN (as a
                 1-D array of length n)
        """
        inp: np.ndarray = self.get_feature_values(x_values_seq)
        ret: List[np.ndarray] = [inp]
        for w in self.weights[:-1]:
            out: np.ndarray = self.dnn_spec.hidden_activation(
                np.dot(inp, w.weights.T)
            )
            if self.dnn_spec.bias:
                inp = np.insert(out, 0, 1., axis=1)
            else:
                inp = out
            ret.append(inp)
        ret.append(
            self.dnn_spec.output_activation(
                np.dot(inp, self.weights[-1].weights.T)
            )[:, 0]
        )
        return ret

    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        return self.forward_propagation(x_values_seq)[-1]

    def backward_propagation(
        self,
        fwd_prop: Sequence[np.ndarray],
        obj_deriv_out: np.ndarray
    ) -> Sequence[np.ndarray]:
        """
        :param fwd_prop represents the result of forward propagation (without
        the final output), a sequence of L 2-D np.ndarrays of the DNN.
        : param obj_deriv_out represents the derivative of the objective
        function with respect to the linear predictor of the final layer.

        :return: list (of length L+1) of |o_l| x |i_l| 2-D arrays,
                 i.e., same as the type of self.weights.weights
        This function computes the gradient (with respect to weights) of
        the objective where the output layer activation function
        is the canonical link function of the conditional distribution of y|x
        """
        deriv: np.ndarray = obj_deriv_out.reshape(1, -1)
        back_prop: List[np.ndarray] = [np.dot(deriv, fwd_prop[-1]) /
                                       deriv.shape[1]]
        # L is the number of hidden layers, n is the number of points
        # layer l deriv represents dObj/ds_l where s_l = i_l . weights_l
        # (s_l is the result of applying layer l without the activation func)
        for i in reversed(range(len(self.weights) - 1)):
            # deriv_l is a 2-D array of dimension |o_l| x n
            # The recursive formulation of deriv is as follows:
            # deriv_{l-1} = (weights_l^T inner deriv_l) haddamard g'(s_{l-1}),
            # which is ((|i_l| x |o_l|) inner (|o_l| x n)) haddamard
            # (|i_l| x n), which is (|i_l| x n) = (|o_{l-1}| x n)
            # Note: g'(s_{l-1}) is expressed as hidden layer activation
            # derivative as a function of o_{l-1} (=i_l).
            deriv = np.dot(self.weights[i + 1].weights.T, deriv) * \
                self.dnn_spec.hidden_activation_deriv(fwd_prop[i + 1].T)
            # If self.dnn_spec.bias is True, then i_l = o_{l-1} + 1, in which
            # case # the first row of the calculated deriv is removed to yield
            # a 2-D array of dimension |o_{l-1}| x n.
            if self.dnn_spec.bias:
                deriv = deriv[1:]
            # layer l gradient is deriv_l inner fwd_prop[l], which is
            # of dimension (|o_l| x n) inner (n x (|i_l|) = |o_l| x |i_l|
            back_prop.append(np.dot(deriv, fwd_prop[i]) / deriv.shape[1])
        return back_prop[::-1]

    def objective_gradient(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        obj_deriv_out_fun: Callable[[Sequence[X], Sequence[float]], np.ndarray]
    ) -> Gradient[DNNApprox[X]]:
        x_vals, y_vals = zip(*xy_vals_seq)
        obj_deriv_out: np.ndarray = obj_deriv_out_fun(x_vals, y_vals)
        fwd_prop: Sequence[np.ndarray] = self.forward_propagation(x_vals)[:-1]
        gradient: Sequence[np.ndarray] = \
            [x + self.regularization_coeff * self.weights[i].weights
             for i, x in enumerate(self.backward_propagation(
                 fwd_prop=fwd_prop,
                 obj_deriv_out=obj_deriv_out
             ))]
        return Gradient(replace(
            self,
            weights=[replace(w, weights=g) for
                     w, g in zip(self.weights, gradient)]
        ))

    def __add__(self, other: DNNApprox[X]) -> DNNApprox[X]:
        return replace(
            self,
            weights=[replace(w, weights=w.weights + o.weights) for
                     w, o in zip(self.weights, other.weights)]
        )

    def __mul__(self, scalar: float) -> DNNApprox[X]:
        return replace(
            self,
            weights=[replace(w, weights=w.weights * scalar)
                     for w in self.weights]
        )

    def update(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]]
    ) -> DNNApprox[X]:
        def deriv_func(x: Sequence[X], y: Sequence[float]) -> np.ndarray:
            return self.dnn_spec.output_activation_deriv(
                np.dot(
                    self.forward_propagation(x)[-2],
                    self.weights[-1].weights.T
                )[:, 0]
            ) * (self.evaluate(x) - np.array(y))

        return self.update_with_gradient(
            self.objective_gradient(xy_vals_seq, deriv_func)
        )

    def update_with_gradient(
        self,
        gradient: Gradient[DNNApprox[X]]
    ) -> DNNApprox[X]:
        return replace(
            self,
            weights=[w.update(g.weights) for w, g in
                     zip(self.weights, gradient.function_approx.weights)]
        )

    def solve(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: Optional[float] = None
    ) -> DNNApprox[X]:
        # Get verbose level from globals if available
        verbose = globals().get('verbose', 0)
        solve_iterations = globals().get('solve_iterations', 10)
        
        # Convert xy_vals_seq to a list for multiple iterations
        result = self
        xy_list = list(xy_vals_seq)
        
        if verbose > 1:
            print(f"    Starting function approximation with {solve_iterations} iterations...")
        
        # Perform multiple update iterations
        for i in range(solve_iterations):
            # Shuffle the data for each epoch
            random.shuffle(xy_list)
            result = result.update(xy_list)
            
            # Show progress every 10% of iterations
            if verbose > 1 and i % max(1, solve_iterations // 10) == 0 and i > 0:
                progress = i / solve_iterations * 100
                print(f"    Function approximation progress: {progress:.1f}% complete", end="\r")
                
        if verbose > 1:
            print("    Function approximation completed.                                                ")
            
        return result


#################################
# Approximate Dynamic Programming #
#################################

# Type aliases
ValueFunctionApprox = FunctionApprox[NonTerminal[S]]
QValueFunctionApprox = FunctionApprox[Tuple[NonTerminal[S], A]]
NTStateDistribution = Distribution[NonTerminal[S]]


def extended_vf(vf: ValueFunctionApprox[S], s: State[S]) -> float:
    return s.on_non_terminal(vf, 0.0)


# Simplified iterate function
def iterate(step: Callable[[X], X], start: X) -> Iterator[X]:
    '''Find the fixed point of a function f by applying it to its own
    result, yielding each intermediate value.
    '''
    state = start
    while True:
        yield state
        state = step(state)


# Simplified converged function
def converged(values: Iterator[X], done: Callable[[X, X], bool]) -> X:
    '''Return the final value when values converge according to done function.'''
    a = next(values)
    for b in values:
        if done(a, b):
            return b
        a = b
    return a


MDP_FuncApproxV_Distribution = Tuple[
    MarkovDecisionProcess[S, A],
    ValueFunctionApprox[S],
    NTStateDistribution[S]
]


def back_opt_vf_and_policy(
    mdp_f0_mu_triples: Sequence[MDP_FuncApproxV_Distribution[S, A]],
    γ: float,
    num_state_samples: int,
    error_tolerance: float
) -> Iterator[Tuple[ValueFunctionApprox[S], DeterministicPolicy[S, A]]]:
    '''Use backwards induction to find the optimal value function and optimal
    policy at each time step, using the given FunctionApprox for each time step
    for a random sample of the time step's states.
    '''
    # Set a global flag to indicate we're in backward induction
    # This will suppress excessive Monte Carlo progress messages
    globals()['_in_backward_induction'] = True
    
    vp: List[Tuple[ValueFunctionApprox[S], DeterministicPolicy[S, A]]] = []

    for i, (mdp, approx0, mu) in enumerate(reversed(mdp_f0_mu_triples)):

        def return_(s_r: Tuple[State[S], float], i=i) -> float:
            s1, r = s_r
            return r + γ * (extended_vf(vp[i-1][0], s1) if i > 0 else 0.)

        this_v = approx0.solve(
            [(s, max(mdp.step(s, a).expectation(return_)
                     for a in mdp.actions(s)))
             for s in mu.sample_n(num_state_samples)],
            error_tolerance
        )

        def deter_policy(state: S) -> A:
            return max(
                ((mdp.step(NonTerminal(state), a).expectation(return_), a)
                 for a in mdp.actions(NonTerminal(state))),
                key=itemgetter(0)
            )[1]

        vp.append((this_v, DeterministicPolicy(deter_policy)))
    
    # Reset the backward induction flag
    globals()['_in_backward_induction'] = False

    return reversed(vp)


MDP_FuncApproxQ_Distribution = Tuple[
    MarkovDecisionProcess[S, A],
    QValueFunctionApprox[S, A],
    NTStateDistribution[S]
]


def back_opt_qvf(
    mdp_f0_mu_triples: Sequence[MDP_FuncApproxQ_Distribution[S, A]],
    γ: float,
    num_state_samples: int,
    error_tolerance: float
) -> Iterator[QValueFunctionApprox[S, A]]:
    '''Use backwards induction to find the optimal q-value function policy at
    each time step, using the given FunctionApprox (for Q-Value) for each time
    step for a random sample of the time step's states.
    '''
    # Set a global flag to indicate we're in backward induction
    # This will suppress excessive Monte Carlo progress messages
    globals()['_in_backward_induction'] = True
    
    horizon: int = len(mdp_f0_mu_triples)
    qvf: List[QValueFunctionApprox[S, A]] = []
    
    # Get verbose level from globals if available
    verbose = globals().get('verbose', 0)
    
    if verbose > 0:
        print("Starting backward induction process...")
        print(f"Horizon: {horizon} time steps")
        print(f"State samples: {num_state_samples}")
        print()
        
    # Start timing the entire process
    start_time_total = time.time()

    for i, (mdp, approx0, mu) in enumerate(reversed(mdp_f0_mu_triples)):
        # Start timing this time step
        start_time_step = time.time()
        
        if verbose > 0:
            time_index = horizon - i - 1
            print(f"Processing time step {time_index} ({i+1}/{horizon})...")
            
        def return_(s_r: Tuple[State[S], float], i=i) -> float:
            s1, r = s_r
            next_return: float = max(
                qvf[i-1]((s1, a)) for a in
                mdp_f0_mu_triples[horizon - i][0].actions(s1)
            ) if i > 0 and isinstance(s1, NonTerminal) else 0.
            return r + γ * next_return
        
        # Generate state-action pairs
        if verbose > 1:
            print(f"  Generating {num_state_samples} state samples...")
            
        states = mu.sample_n(num_state_samples)
        
        # Start timing the expectation calculations
        start_time_expect = time.time()
        
        if verbose > 1:
            print(f"  Computing expectations for state-action pairs...")
            
        xy_vals = []
        total_pairs = num_state_samples * len(list(mdp.actions(states[0])))
        
        for idx, s in enumerate(states):
            for a in mdp.actions(s):
                if verbose > 1 and idx % 5 == 0:
                    progress = (idx * len(list(mdp.actions(s))) + len(list(mdp.actions(s)))) / total_pairs * 100
                    print(f"  Progress: {progress:.1f}% complete", end="\r", flush=True)
                    
                xy_vals.append(((s, a), mdp.step(s, a).expectation(return_)))
        
        # End timing the expectation calculations
        expect_time = time.time() - start_time_expect
        if verbose > 0:
            print(f"  Expectation calculations completed in {expect_time:.2f} seconds.")
        
        # Start timing the function approximation
        start_time_solve = time.time()
        
        if verbose > 1:
            print(f"  Solving function approximation...")
            
        this_qvf = approx0.solve(xy_vals, error_tolerance)
        
        # End timing the function approximation
        solve_time = time.time() - start_time_solve
        if verbose > 0:
            print(f"  Function approximation completed in {solve_time:.2f} seconds.")
            
        qvf.append(this_qvf)
        
        # End timing this time step
        step_time = time.time() - start_time_step
        
        if verbose > 0:
            print(f"  Time step {time_index} completed in {step_time:.2f} seconds.")
            print()

    # End timing the entire process
    total_time = time.time() - start_time_total
    
    if verbose > 0:
        print(f"Backward induction completed in {total_time:.2f} seconds.")
        print()
    
    # Reset the backward induction flag
    globals()['_in_backward_induction'] = False
        
    return reversed(qvf)


#############################
# Asset Allocation Discrete #
#############################

@dataclass(frozen=True)
class AssetAllocDiscrete:
    risky_return_distributions: Sequence[Distribution[float]]
    riskless_returns: Sequence[float]
    utility_func: Callable[[float], float]
    risky_alloc_choices: Sequence[float]
    feature_functions: Sequence[Callable[[Tuple[float, float]], float]]
    dnn_spec: DNNSpec
    initial_wealth_distribution: Distribution[float]

    def time_steps(self) -> int:
        return len(self.risky_return_distributions)

    def uniform_actions(self) -> Choose[float]:
        return Choose(self.risky_alloc_choices)

    def get_mdp(self, t: int) -> MarkovDecisionProcess[float, float]:
        """
        State is Wealth W_t, Action is investment in risky asset (= x_t)
        Investment in riskless asset is W_t - x_t
        """

        distr: Distribution[float] = self.risky_return_distributions[t]
        rate: float = self.riskless_returns[t]
        alloc_choices: Sequence[float] = self.risky_alloc_choices
        steps: int = self.time_steps()
        utility_f: Callable[[float], float] = self.utility_func

        class AssetAllocMDP(MarkovDecisionProcess[float, float]):

            def step(
                self,
                wealth: NonTerminal[float],
                alloc: float
            ) -> SampledDistribution[Tuple[State[float], float]]:

                def sr_sampler_func(
                    wealth=wealth,
                    alloc=alloc
                ) -> Tuple[State[float], float]:
                    next_wealth: float = alloc * (1 + distr.sample()) \
                        + (wealth.state - alloc) * (1 + rate)
                    reward: float = utility_f(next_wealth) \
                        if t == steps - 1 else 0.
                    next_state: State[float] = Terminal(next_wealth) \
                        if t == steps - 1 else NonTerminal(next_wealth)
                    return (next_state, reward)

                return SampledDistribution(
                    sampler=sr_sampler_func,
                    expectation_samples=1000
                )

            def actions(self, wealth: NonTerminal[float]) -> Sequence[float]:
                return alloc_choices

        return AssetAllocMDP()

    def get_qvf_func_approx(self) -> DNNApprox[Tuple[NonTerminal[float], float]]:
        """
        Create a DNN function approximation for Q-Value function
        """
        adam_gradient: AdamGradient = AdamGradient(
            learning_rate=0.1,
            decay1=0.9,
            decay2=0.999
        )
        ffs: List[Callable[[Tuple[NonTerminal[float], float]], float]] = []
        for f in self.feature_functions:
            def this_f(pair: Tuple[NonTerminal[float], float], f=f) -> float:
                return f((pair[0].state, pair[1]))
            ffs.append(this_f)

        return DNNApprox.create(
            feature_functions=ffs,
            dnn_spec=self.dnn_spec,
            adam_gradient=adam_gradient
        )

    def get_states_distribution(self, t: int) -> SampledDistribution[NonTerminal[float]]:
        """
        Create a distribution of states for time t
        """
        actions_distr: Choose[float] = self.uniform_actions()

        def states_sampler_func() -> NonTerminal[float]:
            wealth: float = self.initial_wealth_distribution.sample()
            for i in range(t):
                distr: Distribution[float] = self.risky_return_distributions[i]
                rate: float = self.riskless_returns[i]
                alloc: float = actions_distr.sample()
                wealth = alloc * (1 + distr.sample()) + \
                    (wealth - alloc) * (1 + rate)
            return NonTerminal(wealth)

        return SampledDistribution(states_sampler_func)

    def backward_induction_qvf(self) -> Iterator[QValueFunctionApprox[float, float]]:
        """
        Perform backward induction to find the optimal Q-Value function
        """
        init_fa: DNNApprox[Tuple[NonTerminal[float], float]] = \
            self.get_qvf_func_approx()

        mdp_f0_mu_triples: Sequence[Tuple[
            MarkovDecisionProcess[float, float],
            DNNApprox[Tuple[NonTerminal[float], float]],
            SampledDistribution[NonTerminal[float]]
        ]] = [(
            self.get_mdp(i),
            init_fa,
            self.get_states_distribution(i)
        ) for i in range(self.time_steps())]

        num_state_samples: int = globals().get('state_samples', 300)
        error_tolerance: float = 1e-6

        return back_opt_qvf(
            mdp_f0_mu_triples=mdp_f0_mu_triples,
            γ=1.0,
            num_state_samples=num_state_samples,
            error_tolerance=error_tolerance
        )


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Asset Allocation Discrete Model')
    parser.add_argument('--time-steps', type=int, default=2, help='Number of time steps')
    parser.add_argument('--expectation-samples', type=int, default=1000, help='Number of samples for expectation calculations')
    parser.add_argument('--solve-iterations', type=int, default=10, help='Number of iterations for function approximation')
    parser.add_argument('--state-samples', type=int, default=300, help='Number of state samples for backward induction')
    parser.add_argument('--verbose', type=int, default=0, choices=[0, 1, 2], help='Verbosity level (0=minimal, 1=normal, 2=detailed)')
    parser.add_argument('--mu', type=float, default=0.13, help='Mean of risky return distribution')
    parser.add_argument('--sigma', type=float, default=0.2, help='Standard deviation of risky return distribution')
    parser.add_argument('--rate', type=float, default=0.07, help='Riskless return rate')
    parser.add_argument('--risk-aversion', type=float, default=1.0, help='Risk aversion parameter')
    args = parser.parse_args()

    # Set global parameters
    globals()['verbose'] = args.verbose
    globals()['expectation_samples'] = args.expectation_samples
    globals()['solve_iterations'] = args.solve_iterations
    globals()['state_samples'] = args.state_samples
    globals()['_in_backward_induction'] = False

    # Print parameters
    print("Asset Allocation Discrete Model")
    print("==============================")
    print("Parameters:")
    print(f"  Time steps: {args.time_steps}")
    print(f"  Expectation samples: {args.expectation_samples}")
    print(f"  Solve iterations: {args.solve_iterations}")
    print(f"  State samples: {args.state_samples}")
    print(f"  Verbosity level: {args.verbose}")
    print(f"  μ (risky return mean): {args.mu}")
    print(f"  σ (risky return std): {args.sigma}")
    print(f"  r (riskless return): {args.rate}")
    print(f"  a (risk aversion): {args.risk_aversion}")

    # Calculate base allocation
    excess: float = args.mu - args.rate
    var: float = args.sigma * args.sigma
    base_alloc: float = excess / (args.risk_aversion * var)
    print(f"  Base allocation: {base_alloc:.3f}")
    print()

    # Setup model parameters
    steps: int = args.time_steps
    μ: float = args.mu
    σ: float = args.sigma
    r: float = args.rate
    a: float = args.risk_aversion
    init_wealth: float = 1.0
    init_wealth_stdev: float = 0.1

    risky_ret: Sequence[Gaussian] = [Gaussian(μ=μ, σ=σ) for _ in range(steps)]
    riskless_ret: Sequence[float] = [r for _ in range(steps)]
    utility_function: Callable[[float], float] = lambda x: - np.exp(-a * x) / a
    alloc_choices: Sequence[float] = np.linspace(
        2 / 3 * base_alloc,
        4 / 3 * base_alloc,
        11
    )
    feature_funcs: Sequence[Callable[[Tuple[float, float]], float]] = \
        [
            lambda _: 1.,
            lambda w_x: w_x[0],
            lambda w_x: w_x[1],
            lambda w_x: w_x[1] * w_x[1]
        ]
    dnn: DNNSpec = DNNSpec(
        neurons=[],
        bias=False,
        hidden_activation=lambda x: x,
        hidden_activation_deriv=lambda y: np.ones_like(y),
        output_activation=lambda x: - np.sign(a) * np.exp(-x),
        output_activation_deriv=lambda y: -y
    )
    init_wealth_distr: Gaussian = Gaussian(μ=init_wealth, σ=init_wealth_stdev)

    # Create model
    aad: AssetAllocDiscrete = AssetAllocDiscrete(
        risky_return_distributions=risky_ret,
        riskless_returns=riskless_ret,
        utility_func=utility_function,
        risky_alloc_choices=alloc_choices,
        feature_functions=feature_funcs,
        dnn_spec=dnn,
        initial_wealth_distribution=init_wealth_distr
    )

    # Run backward induction
    it_qvf: Iterator[QValueFunctionApprox[float, float]] = \
        aad.backward_induction_qvf()

    # Print results
    print("Backward Induction on Q-Value Function")
    print("--------------------------------------")
    print()
    for t, q in enumerate(it_qvf):
        print(f"Time {t:d}")
        print()
        opt_alloc: float = max(
            ((q((NonTerminal(init_wealth), ac)), ac) for ac in alloc_choices),
            key=itemgetter(0)
        )[1]
        val: float = max(q((NonTerminal(init_wealth), ac))
                         for ac in alloc_choices)
        print(f"Opt Risky Allocation = {opt_alloc:.3f}, Opt Val = {val:.3f}")
        print("Optimal Weights below:")
        for wts in q.weights:
            print(wts.weights)
        print()

    # Print analytical solution
    print("Analytical Solution")
    print("-------------------")
    print()

    for t in range(steps):
        print(f"Time {t:d}")
        print()
        left: int = steps - t
        growth: float = (1 + r) ** (left - 1)
        alloc: float = base_alloc / growth
        vval: float = - np.exp(- excess * excess * left / (2 * var)
                               - a * growth * (1 + r) * init_wealth) / a
        bias_wt: float = excess * excess * (left - 1) / (2 * var) + \
            np.log(np.abs(a))
        w_t_wt: float = a * growth * (1 + r)
        x_t_wt: float = a * excess * growth
        x_t2_wt: float = - var * (a * growth) ** 2 / 2

        print(f"Opt Risky Allocation = {alloc:.3f}, Opt Val = {vval:.3f}")
        print(f"Bias Weight = {bias_wt:.3f}")
        print(f"W_t Weight = {w_t_wt:.3f}")
        print(f"x_t Weight = {x_t_wt:.3f}")
        print(f"x_t^2 Weight = {x_t2_wt:.3f}")
        print()
