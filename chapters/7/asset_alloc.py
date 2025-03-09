"""
Asset Allocation Discrete model using the RL library.

This script implements the Asset Allocation Discrete model using the components
from the rl_lib package instead of containing all the code.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, Iterator

from rl_lib.distribution import Gaussian, Choose, Constant
from rl_lib.mdp import State, Terminal, NonTerminal, MarkovDecisionProcess, DeterministicPolicy
from rl_lib.function_approx import DNNSpec, DNNApprox, AdamGradient
from rl_lib.adp import back_opt_vf_and_policy, back_opt_qvf

@dataclass(frozen=True)
class AssetAllocDiscrete:
    """
    Asset allocation model with discrete action choices.
    
    This model determines optimal allocation between risky and riskless assets
    over multiple time steps.
    """
    risky_return_distributions: Sequence[Gaussian]  # Distribution of risky asset returns
    riskless_returns: Sequence[float]  # Risk-free returns
    utility_func: Callable[[float], float]  # Utility function
    risky_alloc_choices: Sequence[float]  # Possible allocations to risky asset
    feature_functions: Sequence[Callable]  # Feature functions for function approximation
    dnn_spec: DNNSpec  # Neural network specification
    initial_wealth_distribution: Gaussian  # Distribution of initial wealth
    
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
            def actions(self, wealth: NonTerminal) -> Sequence[float]:
                """Return available actions (allocation choices)."""
                return alloc_choices
            
            def step(self, wealth: NonTerminal, alloc: float):
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
                
                from rl_lib.distribution import SampledDistribution
                return SampledDistribution(sampler=sr_sampler_func, expectation_samples=1000)
        
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
    
    def get_states_distribution(self, t: int):
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
        
        from rl_lib.distribution import SampledDistribution
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
            key=lambda x: x[0]
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
