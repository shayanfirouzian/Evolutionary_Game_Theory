import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Initial parameter setting
N = 100  # Total players
r = 4  # Payoff coefficient in public goods games
G = 5  # Number of players per group
delta = 0.6  # Nonlinear payoff coefficient
p = 0.5  # The probability of a synergy group
x0 = 0.386  # Initial cooperator ratio
s = 0.1  # Sensitivity parameters of Fermi function

# Initial strategy assignment, 1 for cooperation, 0 for defection
strategies = np.random.choice([1, 0], size=N, p=[x0, 1 - x0])

T = 100000000  # Time steps
cooperator_ratios = []  # Stores the proportion of cooperators per time step


def calculate_payoff(n_C, synergy):
    """Calculate the payoff for each player based on the number of cooperators and group type (synergy or discounting)"""
    if synergy:
        omega = 1 + delta
    else:
        omega = 1 - delta

    return (r / G) * (1 - (omega) ** n_C) / (1 - omega)


for t in range(T):
    i = np.random.choice(N)  # Choose a player i at random
    n_C = np.sum(np.random.choice(strategies,
                                  G - 1))  # G-1 neighbors are randomly selected and the number of cooperators is calculated
    synergy = np.random.rand() < p  # Decide if it is a synergy group
    payoffs_1 = calculate_payoff(n_C + strategies[i], synergy)
    if strategies[i] == 1:
        payoffs_1 -= 1  # If the current player is a cooperator, subtract the cost of their contribution

    j = np.random.choice(N)  # Choose a random player to compare with

    if strategies[i] != strategies[j]:  # Learn if the strategies are different
        n_C = np.sum(np.random.choice(strategies, G - 1))
        synergy = np.random.rand() < p
        payoffs_2 = calculate_payoff(n_C + strategies[j], synergy)
        if strategies[j] == 1:
            payoffs_2 -= 1

        pi_diff = payoffs_2 - payoffs_1
        probability = 1 / (1 + np.exp(-s * pi_diff))  # Calculate the probability according to Fermi's rules
        if np.random.rand() < probability:
            strategies[i] = strategies[j]  # i adopts j's strategy

    # Calculate and store the cooperator ratio
    cooperator_ratio = np.mean(strategies)
    cooperator_ratios.append(cooperator_ratio)

    # Check if all players are cooperators or defectors
    if cooperator_ratio == 0 or cooperator_ratio == 1:
        while len(cooperator_ratios) < T:
            cooperator_ratios.append(cooperator_ratio)
        break  # If full cooperation or defection is achieved, stop the simulation

# plot figure
plt.figure(figsize=(8, 6))
plt.plot(cooperator_ratios, label='Cooperator ratio', color='blue')
plt.xscale('log')  # مقیاس لگاریتمی برای محور x
plt.xlabel('Time')
plt.ylim(0, 1)
plt.ylabel('Cooperator ratio')
plt.title('Evolution of cooperator ratio over time')

# add parameters
params_text = (
    f"N = {N}\n"
    f"r = {r}\n"
    f"G = {G}\n"
    f"delta = {delta}\n"
    f"p = {p}\n"
    f"x0 = {x0}\n"
    f"s = {s}"
)

# show parameters in box
plt.text(0.05, 0.6, params_text, fontsize=10, verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.8), transform=plt.gca().transAxes)

plt.legend()
plt.show()