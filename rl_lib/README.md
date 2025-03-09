# RL Library

A modular reinforcement learning library for Python, designed with a focus on readability and extensibility.

## Overview

This library provides components for building and solving reinforcement learning problems, including:

- Probability distributions
- Markov Decision Processes (MDPs)
- Function approximation (including neural networks)
- Approximate Dynamic Programming (ADP) algorithms

## Installation

```bash
# Install from the local directory
pip install -e .
```

## Modules

### Distribution

The `distribution` module provides classes for representing probability distributions:

- `Distribution`: Base class for all distributions
- `SampledDistribution`: Distribution defined by a sampling function
- `Gaussian`: Normal distribution with given mean and standard deviation
- `Choose`: Uniform distribution over a finite set of options
- `Constant`: Distribution with a single outcome that has probability 1

### MDP

The `mdp` module provides classes for representing Markov Decision Processes:

- `State`, `Terminal`, `NonTerminal`: Classes for representing states
- `MarkovDecisionProcess`: Base class for MDPs
- `Policy`, `DeterministicPolicy`: Classes for representing policies

### Function Approximation

The `function_approx` module provides classes for function approximation:

- `FunctionApprox`: Abstract base class for function approximation
- `Gradient`: Gradient of a function approximation
- `AdamGradient`: Parameters for Adam gradient descent optimizer
- `Weights`: Weights for a neural network with Adam optimizer state
- `DNNSpec`: Specification for a neural network architecture
- `DNNApprox`: Neural network function approximation

### Utils

The `utils` module provides utility functions:

- `iterate`: Generate a sequence by repeatedly applying a function
- `converge`: Read from an iterator until convergence
- `converged`: Return the final value when an iterator converges
- `accumulate`: Make an iterator that returns accumulated results

### ADP

The `adp` module provides functions for Approximate Dynamic Programming:

- `back_opt_vf_and_policy`: Find optimal value function and policy using backward induction
- `back_opt_qvf`: Find optimal Q-value function using backward induction

## Example

Here's a simple example of using the library to solve an asset allocation problem:

```python
from rl_lib.distribution import Gaussian
from rl_lib.function_approx import DNNSpec
from rl_lib.mdp import NonTerminal

# Define model parameters
μ = 0.13  # Mean return of risky asset
σ = 0.2   # Standard deviation of risky asset return
r = 0.07  # Risk-free rate

# Create a Gaussian distribution for risky returns
risky_return = Gaussian(μ=μ, σ=σ)

# Define a neural network specification
dnn = DNNSpec(
    neurons=[],  # No hidden layers (linear model)
    bias=False,
    hidden_activation=lambda x: x,  # Identity
    hidden_activation_deriv=lambda y: np.ones_like(y),
    output_activation=lambda x: x,  # Identity
    output_activation_deriv=lambda y: np.ones_like(y)
)

# ... (create MDP, solve using backward induction)
```

For a complete example, see the `asset_alloc_discrete_lib.py` file in the `chapters/7` directory.

## License

MIT
