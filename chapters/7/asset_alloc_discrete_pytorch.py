"""
PyTorch implementation of the Asset Allocation Discrete model.
This script replaces the custom neural network implementation with PyTorch.

NOTE: This version is currently broken and needs further debugging.
Please use asset_alloc_discrete_standalone.py instead for a working implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace, field
from typing import (Callable, Dict, Generic, Iterator, Iterable, List, 
                   Mapping, Optional, Sequence, Tuple, TypeVar)
from operator import itemgetter
import numpy as np
import itertools
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Type variables
A = TypeVar('A')
S = TypeVar('S')
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
    def expectation(
        self,
        f: Callable[[A], float]
    ) -> float:
        '''Return the expecation of f(X) where X is the
        random variable for the distribution and f is an
        arbitrary function from X to float
        '''
        pass

    def apply(
        self,
        f: Callable[[A], Distribution[B]]
    ) -> Distribution[B]:
        '''Apply a function that returns a distribution to the outcomes of
        this distribution. This lets us express *dependent random
        variables*.
        '''
        def sample():
            a = self.sample()
            b_dist = f(a)
            return b_dist.sample()

        return SampledDistribution(sample)


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

    def expectation(
        self,
        f: Callable[[A], float]
    ) -> float:
        '''Return a sampled approximation of the expectation of f(X) for some f.'''
        return sum(f(self.sample()) for _ in
                   range(self.expectation_samples)) / self.expectation_samples


class Gaussian(SampledDistribution[float]):
    '''A Gaussian distribution with the given μ and σ.'''

    μ: float
    σ: float

    def __init__(self, μ: float, σ: float, expectation_samples: int = 1000):
        self.μ = μ
        self.σ = σ
        super().__init__(
            sampler=lambda: np.random.normal(loc=self.μ, scale=self.σ),
            expectation_samples=expectation_samples
        )


class Choose(Distribution[A]):
    '''Select an element of the given list uniformly at random.'''

    options: Sequence[A]

    def __init__(self, options: Iterable[A]):
        self.options = list(options)

    def sample(self) -> A:
        import random
        return random.choice(self.options)

    def expectation(self, f: Callable[[A], float]) -> float:
        '''Calculate expectation by averaging over all possible outcomes.'''
        return sum(f(option) for option in self.options) / len(self.options)


@dataclass(frozen=True)
class Constant(Distribution[A]):
    '''A distribution that has a single outcome with probability 1.'''
    value: A

    def sample(self) -> A:
        return self.value

    def expectation(self, f: Callable[[A], float]) -> float:
        return f(self.value)


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

    def act(self, state: NonTerminal[S]) -> Constant[A]:
        return Constant(self.action_for(state.state))


###############
# Iterate.py #
###############

def iterate(step: Callable[[X], X], start: X) -> Iterator[X]:
    '''Find the fixed point of a function f by applying it to its own
    result, yielding each intermediate value.

    That is, for a function f, iterate(f, x) will give us a generator
    producing:

    x, f(x), f(f(x)), f(f(f(x)))...
    '''
    state = start

    while True:
        yield state
        state = step(state)


def last(values: Iterator[X]) -> Optional[X]:
    '''Return the last value of the given iterator.

    Returns None if the iterator is empty.

    If the iterator does not end, this function will loop forever.
    '''
    try:
        *_, last_element = values
        return last_element
    except ValueError:
        return None


def converge(values: Iterator[X], done: Callable[[X, X], bool]) -> Iterator[X]:
    '''Read from an iterator until two consecutive values satisfy the
    given done function or the input iterator ends.

    Raises an error if the input iterator is empty.

    Will loop forever if the input iterator doesn't end *or* converge.
    '''
    a = next(values, None)
    if a is None:
        return

    yield a

    for b in values:
        yield b
        if done(a, b):
            return

        a = b


def converged(values: Iterator[X],
              done: Callable[[X, X], bool]) -> X:
    '''Return the final value of the given iterator when its values
    converge according to the done function.

    Raises an error if the iterator is empty.

    Will loop forever if the input iterator doesn't end *or* converge.
    '''
    result = last(converge(values, done))

    if result is None:
        raise ValueError("converged called on an empty iterator")

    return result


def accumulate(
        iterable: Iterable[X],
        func: Callable[[Y, X], Y],
        *,
        initial: Optional[Y]
) -> Iterator[Y]:
    '''Make an iterator that returns accumulated sums, or accumulated
    results of other binary functions (specified via the optional func
    argument).

    If func is supplied, it should be a function of two
    arguments. Elements of the input iterable may be any type that can
    be accepted as arguments to func. (For example, with the default
    operation of addition, elements may be any addable type including
    Decimal or Fraction.)

    Usually, the number of elements output matches the input
    iterable. However, if the keyword argument initial is provided,
    the accumulation leads off with the initial value so that the
    output has one more element than the input iterable.
    '''
    if initial is not None:
        iterable = itertools.chain([initial], iterable)  # type: ignore

    return itertools.accumulate(iterable, func)  # type: ignore


#######################
# PyTorch Model      #
#######################

class CustomActivation(nn.Module):
    """Custom activation function for the output layer"""
    def __init__(self, a: float):
        super().__init__()
        self.a = a
        
    def forward(self, x):
        return -torch.sign(torch.tensor(self.a)) * torch.exp(-x)


class DNNModel(nn.Module):
    """PyTorch implementation of the DNN model"""
    def __init__(self, input_dim: int, a: float):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
        self.output_activation = CustomActivation(a)
        
        # Initialize weights similar to the original implementation
        nn.init.normal_(self.linear.weight, std=1.0/np.sqrt(input_dim))
        
    def forward(self, x):
        x = self.linear(x)
        x = self.output_activation(x)
        return x.squeeze()


class PyTorchApprox(Generic[X]):
    """PyTorch implementation of the function approximation"""
    
    def __init__(
        self,
        feature_functions: Sequence[Callable[[X], float]],
        model: nn.Module,
        optimizer: optim.Optimizer,
        regularization_coeff: float = 0.0
    ):
        self.feature_functions = feature_functions
        self.model = model
        self.optimizer = optimizer
        self.regularization_coeff = regularization_coeff
        
    def get_feature_values(self, x_values_seq: Iterable[X]) -> torch.Tensor:
        """Convert input values to feature tensor"""
        features = [[f(x) for f in self.feature_functions] for x in x_values_seq]
        return torch.tensor(features, dtype=torch.float32)
        
    def evaluate(self, x_values_seq: Iterable[X]) -> torch.Tensor:
        """Evaluate the model on the given inputs"""
        features = self.get_feature_values(x_values_seq)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features)
        return outputs
        
    def __call__(self, x_value: X) -> float:
        """Evaluate the model on a single input"""
        result = self.evaluate([x_value])
        return float(result.item() if result.numel() == 1 else result[0].item())
    
    def objective_gradient(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        obj_deriv_out_fun: Callable[[Sequence[X], Sequence[float]], torch.Tensor]
    ) -> 'Gradient[PyTorchApprox[X]]':
        """Compute the gradient of the objective function"""
        x_vals, y_vals = zip(*xy_vals_seq)
        
        # Get features and convert to tensor
        features = self.get_feature_values(x_vals)
        
        # Forward pass to get predictions
        self.model.train()
        outputs = self.model(features)
        
        # Get the derivative of the objective function
        obj_deriv_out = obj_deriv_out_fun(x_vals, y_vals)
        
        # Create a copy of the model with zero weights for the gradient
        gradient_model = DNNModel(input_dim=len(self.feature_functions), a=self.model.output_activation.a)
        
        # Zero out the weights in the gradient model
        for param in gradient_model.parameters():
            param.data.zero_()
        
        # Create a new optimizer for the gradient model
        gradient_optimizer = optim.Adam(
            gradient_model.parameters(),
            lr=self.optimizer.param_groups[0]['lr'],
            betas=self.optimizer.param_groups[0]['betas']
        )
        
        # Compute gradients
        outputs.backward(obj_deriv_out)
        
        # Copy gradients to the gradient model's weights
        for param, grad_param in zip(self.model.parameters(), gradient_model.parameters()):
            if param.grad is not None:
                grad_param.data = param.grad.data.clone()
                
                # Add regularization if needed
                if self.regularization_coeff > 0:
                    grad_param.data += self.regularization_coeff * param.data
        
        # Create a new PyTorchApprox with the gradient model
        return Gradient(PyTorchApprox(
            feature_functions=self.feature_functions,
            model=gradient_model,
            optimizer=gradient_optimizer,
            regularization_coeff=self.regularization_coeff
        ))
    
    def update_with_gradient(
        self,
        gradient: 'Gradient[PyTorchApprox[X]]'
    ) -> 'PyTorchApprox[X]':
        """Update the model using the gradient"""
        # Create a new model with the same architecture
        new_model = DNNModel(input_dim=len(self.feature_functions), a=self.model.output_activation.a)
        
        # Copy the current weights
        for param, new_param in zip(self.model.parameters(), new_model.parameters()):
            new_param.data = param.data.clone()
        
        # Create a new optimizer
        new_optimizer = optim.Adam(
            new_model.parameters(),
            lr=self.optimizer.param_groups[0]['lr'],
            betas=self.optimizer.param_groups[0]['betas']
        )
        
        # Copy optimizer state
        for i, (param_group, new_param_group) in enumerate(zip(self.optimizer.param_groups, new_optimizer.param_groups)):
            for param, new_param in zip(param_group['params'], new_param_group['params']):
                # Copy state if it exists
                if param in self.optimizer.state:
                    new_optimizer.state[new_param] = {
                        key: value.clone() if torch.is_tensor(value) else value
                        for key, value in self.optimizer.state[param].items()
                    }
        
        # Apply the gradient update manually
        for param, grad_param in zip(new_model.parameters(), gradient.function_approx.model.parameters()):
            # Get optimizer state for this parameter
            if param in new_optimizer.state:
                state = new_optimizer.state[param]
                
                # Update exponential moving averages
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                beta1, beta2 = new_optimizer.param_groups[0]['betas']
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad_param.data, alpha=1 - beta1)
                
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad_param.data, grad_param.data, value=1 - beta2)
                
                # Get step size
                step_size = new_optimizer.param_groups[0]['lr']
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** (state['step'] + 1)
                bias_correction2 = 1 - beta2 ** (state['step'] + 1)
                
                # Compute the step
                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                
                # Update parameters
                param.data.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(1e-8), value=-step_size)
                
                # Increment step
                state['step'] += 1
            else:
                # Initialize state and apply first update
                new_optimizer.step()
        
        # Return a new PyTorchApprox with the updated model
        return PyTorchApprox(
            feature_functions=self.feature_functions,
            model=new_model,
            optimizer=new_optimizer,
            regularization_coeff=self.regularization_coeff
        )
    
    def update(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]]
    ) -> 'PyTorchApprox[X]':
        """Update the model with new data"""
        def deriv_func(x: Sequence[X], y: Sequence[float]) -> torch.Tensor:
            y_tensor = torch.tensor(y, dtype=torch.float32)
            return self.evaluate(x) - y_tensor
        
        return self.update_with_gradient(
            self.objective_gradient(xy_vals_seq, deriv_func)
        )
    
    def solve(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: Optional[float] = None
    ) -> 'PyTorchApprox[X]':
        """Train the model using the same approach as the original implementation"""
        tol: float = 1e-6 if error_tolerance is None else error_tolerance
        
        def done(
            a: 'PyTorchApprox[X]',
            b: 'PyTorchApprox[X]',
            tol: float = tol
        ) -> bool:
            return a.within(b, tol)
        
        return converged(
            self.iterate_updates(
                itertools.repeat(list(xy_vals_seq))
            ),
            done=done
        )
    
    def iterate_updates(
        self,
        xy_seq_stream: Iterator[Iterable[Tuple[X, float]]]
    ) -> Iterator['PyTorchApprox[X]']:
        """Iterate updates using the same approach as the original implementation"""
        return accumulate(
            xy_seq_stream,
            lambda fa, xy: fa.update(xy),
            initial=self
        )
    
    def __add__(self, other: 'PyTorchApprox[X]') -> 'PyTorchApprox[X]':
        """Add two function approximations"""
        # Create a new model with the same architecture
        new_model = DNNModel(input_dim=len(self.feature_functions), a=self.model.output_activation.a)
        
        # Add the weights
        for param, other_param, new_param in zip(
            self.model.parameters(), 
            other.model.parameters(), 
            new_model.parameters()
        ):
            new_param.data = param.data + other_param.data
        
        # Create a new optimizer
        new_optimizer = optim.Adam(
            new_model.parameters(),
            lr=self.optimizer.param_groups[0]['lr'],
            betas=self.optimizer.param_groups[0]['betas']
        )
        
        # Return a new PyTorchApprox
        return PyTorchApprox(
            feature_functions=self.feature_functions,
            model=new_model,
            optimizer=new_optimizer,
            regularization_coeff=self.regularization_coeff
        )
    
    def __mul__(self, scalar: float) -> 'PyTorchApprox[X]':
        """Multiply by a scalar"""
        # Create a new model with the same architecture
        new_model = DNNModel(input_dim=len(self.feature_functions), a=self.model.output_activation.a)
        
        # Multiply the weights by the scalar
        for param, new_param in zip(self.model.parameters(), new_model.parameters()):
            new_param.data = param.data * scalar
        
        # Create a new optimizer
        new_optimizer = optim.Adam(
            new_model.parameters(),
            lr=self.optimizer.param_groups[0]['lr'],
            betas=self.optimizer.param_groups[0]['betas']
        )
        
        # Return a new PyTorchApprox
        return PyTorchApprox(
            feature_functions=self.feature_functions,
            model=new_model,
            optimizer=new_optimizer,
            regularization_coeff=self.regularization_coeff
        )
    
    def within(self, other: 'PyTorchApprox[X]', tolerance: float) -> bool:
        """Check if this model is within tolerance of another model"""
        for p1, p2 in zip(self.model.parameters(), other.model.parameters()):
            if not torch.all(torch.abs(p1 - p2) <= tolerance):
                return False
        return True


@dataclass(frozen=True)
class Gradient(Generic[F]):
    function_approx: F

    def __add__(self, x: 'Gradient[F]') -> 'Gradient[F]':
        return Gradient(self.function_approx + x.function_approx)

    def __mul__(self: 'Gradient[F]', x: float) -> 'Gradient[F]':
        return Gradient(self.function_approx * x)


#################################
# Approximate Dynamic Programming #
#################################

# Type aliases
ValueFunctionApprox = PyTorchApprox[NonTerminal[S]]
QValueFunctionApprox = PyTorchApprox[Tuple[NonTerminal[S], A]]
NTStateDistribution = Distribution[NonTerminal[S]]


def extended_vf(vf: ValueFunctionApprox[S], s: State[S]) -> float:
    return s.on_non_terminal(vf, 0.0)


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
    horizon: int = len(mdp_f0_mu_triples)
    qvf: List[QValueFunctionApprox[S, A]] = []

    for i, (mdp, approx0, mu) in enumerate(reversed(mdp_f0_mu_triples)):

        def return_(s_r: Tuple[State[S], float], i=i) -> float:
            s1, r = s_r
            next_return: float = max(
                qvf[i-1]((s1, a)) for a in
                mdp_f0_mu_triples[horizon - i][0].actions(s1)
            ) if i > 0 and isinstance(s1, NonTerminal) else 0.
            return r + γ * next_return

        this_qvf = approx0.solve(
            [((s, a), mdp.step(s, a).expectation(return_))
             for s in mu.sample_n(num_state_samples) for a in mdp.actions(s)],
            error_tolerance
        )

        qvf.append(this_qvf)

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
    a: float  # Risk aversion parameter
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

    def get_qvf_func_approx(self) -> \
            PyTorchApprox[Tuple[NonTerminal[float], float]]:
        """Create a PyTorch Q-value function approximation"""
        
        # Create feature functions for the Q-value function
        ffs: List[Callable[[Tuple[NonTerminal[float], float]], float]] = []
        for f in self.feature_functions:
            def this_f(pair: Tuple[NonTerminal[float], float], f=f) -> float:
                return f((pair[0].state, pair[1]))
            ffs.append(this_f)
        
        # Create PyTorch model
        model = DNNModel(input_dim=len(ffs), a=self.a)
        
        # Create Adam optimizer with the same parameters as the original
        optimizer = optim.Adam(
            model.parameters(),
            lr=0.1,  # Original learning_rate
            betas=(0.9, 0.999)  # Original decay1, decay2
        )
        
        return PyTorchApprox(
            feature_functions=ffs,
            model=model,
            optimizer=optimizer
        )

    def get_states_distribution(self, t: int) -> \
            SampledDistribution[NonTerminal[float]]:

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

    def backward_induction_qvf(self) -> \
            Iterator[QValueFunctionApprox[float, float]]:

        init_fa: PyTorchApprox[Tuple[NonTerminal[float], float]] = \
            self.get_qvf_func_approx()

        mdp_f0_mu_triples: Sequence[Tuple[
            MarkovDecisionProcess[float, float],
            PyTorchApprox[Tuple[NonTerminal[float], float]],
            SampledDistribution[NonTerminal[float]]
        ]] = [(
            self.get_mdp(i),
            init_fa,  # Use the same function approximation for all time steps
            self.get_states_distribution(i)
        ) for i in range(self.time_steps())]

        num_state_samples: int = 300
        error_tolerance: float = 1e-6

        return back_opt_qvf(
            mdp_f0_mu_triples=mdp_f0_mu_triples,
            γ=1.0,
            num_state_samples=num_state_samples,
            error_tolerance=error_tolerance
        )

    def get_vf_func_approx(
        self,
        ff: Sequence[Callable[[NonTerminal[float]], float]]
    ) -> PyTorchApprox[NonTerminal[float]]:
        """Create a PyTorch value function approximation"""
        
        # Create PyTorch model
        model = DNNModel(input_dim=len(ff), a=self.a)
        
        # Create Adam optimizer with the same parameters as the original
        optimizer = optim.Adam(
            model.parameters(),
            lr=0.1,  # Original learning_rate
            betas=(0.9, 0.999)  # Original decay1, decay2
        )
        
        return PyTorchApprox(
            feature_functions=ff,
            model=model,
            optimizer=optimizer
        )

    def backward_induction_vf_and_pi(
        self,
        ff: Sequence[Callable[[NonTerminal[float]], float]]
    ) -> Iterator[Tuple[ValueFunctionApprox[float],
                        DeterministicPolicy[float, float]]]:

        init_fa: PyTorchApprox[NonTerminal[float]] = self.get_vf_func_approx(ff)

        mdp_f0_mu_triples: Sequence[Tuple[
            MarkovDecisionProcess[float, float],
            PyTorchApprox[NonTerminal[float]],
            SampledDistribution[NonTerminal[float]]
        ]] = [(
            self.get_mdp(i),
            init_fa,  # Use the same function approximation for all time steps
            self.get_states_distribution(i)
        ) for i in range(self.time_steps())]

        num_state_samples: int = 300
        error_tolerance: float = 1e-8

        return back_opt_vf_and_policy(
            mdp_f0_mu_triples=mdp_f0_mu_triples,
            γ=1.0,
            num_state_samples=num_state_samples,
            error_tolerance=error_tolerance
        )


if __name__ == '__main__':
    from pprint import pprint

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    steps: int = 4
    μ: float = 0.13
    σ: float = 0.2
    r: float = 0.07
    a: float = 1.0
    init_wealth: float = 1.0
    init_wealth_stdev: float = 0.1

    excess: float = μ - r
    var: float = σ * σ
    base_alloc: float = excess / (a * var)

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

    init_wealth_distr: Gaussian = Gaussian(μ=init_wealth, σ=init_wealth_stdev)

    aad: AssetAllocDiscrete = AssetAllocDiscrete(
        risky_return_distributions=risky_ret,
        riskless_returns=riskless_ret,
        utility_func=utility_function,
        risky_alloc_choices=alloc_choices,
        feature_functions=feature_funcs,
        a=a,
        initial_wealth_distribution=init_wealth_distr
    )

    # vf_ff: Sequence[Callable[[NonTerminal[float]], float]] = [lambda _: 1., lambda w: w.state]
    # it_vf: Iterator[Tuple[PyTorchApprox[NonTerminal[float]], DeterministicPolicy[float, float]]] = \
    #     aad.backward_induction_vf_and_pi(vf_ff)

    # print("Backward Induction: VF And Policy")
    # print("---------------------------------")
    # print()
    # for t, (v, p) in enumerate(it_vf):
    #     print(f"Time {t:d}")
    #     print()
    #     opt_alloc: float = p.action_for(init_wealth)
    #     val: float = v(NonTerminal(init_wealth))
    #     print(f"Opt Risky Allocation = {opt_alloc:.2f}, Opt Val = {val:.3f}")
    #     print("Weights")
    #     for param in v.model.parameters():
    #         print(param.data)
    #     print()

    it_qvf: Iterator[QValueFunctionApprox[float, float]] = \
        aad.backward_induction_qvf()

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
        for param in q.model.parameters():
            pprint(param.data.numpy())
        print()

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
