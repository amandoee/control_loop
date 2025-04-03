import numpy as np

# Simulation parameters
num_simulations = 10000  # Number of simulations
starting_bankroll = 500  # Initial amount in dollars
bet_size = 10  # Fixed bet amount per round
num_rounds = 500  # Number of rounds
min_survival_amount = 20  # Survival threshold

# Probabilities
p_win = 18 / 37  # Probability of winning a red bet
p_loss = 19 / 37  # Probability of losing a red bet
win_amount = 10 # Profit when winning
loss_amount = -10  # Loss when losing

# Run simulations
final_amounts = []
survival_count = 0

for _ in range(num_simulations):
    bankroll = starting_bankroll

    for _ in range(num_rounds):
        if bankroll < bet_size:  # Stop if can't afford a bet
            break
        outcome = np.random.choice([win_amount, loss_amount], p=[p_win, p_loss])
        bankroll += outcome

    final_amounts.append(bankroll)

    if bankroll > min_survival_amount:
        survival_count += 1

# Compute results
average_final_amount = np.mean(final_amounts)
survival_probability = survival_count / num_simulations

print(average_final_amount, survival_probability)
