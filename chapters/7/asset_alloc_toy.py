import math
import random
import numpy as np
import matplotlib.pyplot as plt

# -- 1. Problem Setup --
T = 3
# We'll assume each step's risky asset return ~ Normal(mu=0.13, sigma=0.20)
mu = 0.13
sigma = 0.20
riskless_return = 0.07

# Risk aversion parameter
a = 1.0

# Calculate base allocation using analytical formula
excess = mu - riskless_return
var = sigma * sigma
base_alloc = excess / (a * var)

# Discrete set of possible actions (risky-asset allocations)
possible_actions = np.linspace(0.0, 2.0, 25)  # e.g. from 0 up to 2 units of wealth

# Utility at final time:
def utility_final(wealth):
    # CARA utility with risk aversion parameter a
    return -np.exp(-a * wealth) / a

# Q-values stored in an array or dictionary: Q[t][wealth_index][action_index]
# Here we do a simplified "grid over wealth" approach; in real code you'd use a NN
grid_wealth = np.linspace(0.0, 2.0, 21)  # For demonstration
Q = [ np.zeros((len(grid_wealth), len(possible_actions))) for _ in range(T) ]

def next_wealth(W, x):
    # Sample from the risky asset return
    r_risky = random.gauss(mu, sigma)
    return x*(1 + r_risky) + (W - x)*(1 + riskless_return)

# -- 2. Backward Induction in a Simple Grid + Monte Carlo Way --
N_samples = 2000  # how many Monte Carlo samples per state-action

for w_idx, W in enumerate(grid_wealth):
    for a_idx, x in enumerate(possible_actions):
        # At final step t=2, Q^*(W, x) = E[ Utility( next wealth ) ]
        # We'll approximate by sampling from the distribution
        sum_util = 0.0
        for _ in range(N_samples):
            W_next = next_wealth(W, x)
            sum_util += utility_final(W_next)
        Q[2][w_idx, a_idx] = sum_util / N_samples

# Now for t=1 and t=0
for t in [1, 0]:
    for w_idx, W in enumerate(grid_wealth):
        for a_idx, x in enumerate(possible_actions):
            # Next wealth distribution from W, x
            # Then Q^*_t(W, x) = E[ max_{x'} Q^*_{t+1}( nextWealth, x') ]
            sum_val = 0.0
            for _ in range(N_samples):
                W_next = next_wealth(W, x)
                # find best action x' at t+1
                # We'll look up Q[t+1][W_next_index, x']
                # but W_next might not align exactly with grid...
                # so do a quick linear interpolation or just pick nearest
                w_next_idx = np.argmin(np.abs(grid_wealth - W_next))
                # best Q over all actions:
                best_val = np.max(Q[t+1][w_next_idx, :])
                sum_val += best_val
            Q[t][w_idx, a_idx] = sum_val / N_samples

# -- 3. Extract the (approx) optimal policy for each t and wealth
optimal_actions = {}
for t in range(T):
    best_action_for_wealth = []
    for w_idx, W in enumerate(grid_wealth):
        # pick the argmax of Q[t][w_idx, :]
        a_idx_best = np.argmax(Q[t][w_idx, :])
        best_action_for_wealth.append(possible_actions[a_idx_best])
    optimal_actions[t] = best_action_for_wealth

# -- 4. Plot: "Optimal Risky Allocation" vs "Wealth" for each time t
for t in range(T):
    plt.plot(grid_wealth, optimal_actions[t], label=f"Time t={t}")

plt.xlabel("Current Wealth")
plt.ylabel("Optimal Risky Allocation")
plt.title("Optimal Policy vs. Wealth (Discrete-Time Backward Induction)")
plt.legend()
plt.show()
