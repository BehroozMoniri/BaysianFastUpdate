import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

#
# class OnlineBinomialModel:
#     def __init__(self, alpha_prior=1.0, beta_prior=1.0):
#         # Initialize with Beta prior parameters (1,1 is a uniform prior)
#         self.alpha = alpha_prior
#         self.beta = beta_prior
#
#     def update(self, successes, trials):
#         """
#         Update the posterior with new data.
#         - successes: number of successful events (e.g., clicks)
#         - trials: total number of events (e.g., impressions)
#         """
#         failures = trials - successes
#         self.alpha += successes
#         self.beta += failures
#
#     def get_expected_probability(self):
#         # The mean of a Beta(alpha, beta) distribution is alpha / (alpha + beta)
#         return self.alpha / (self.alpha + self.beta)
#
#     def get_credible_interval(self, confidence=0.95):
#         lower_prob = (1.0 - confidence) / 2.0
#         upper_prob = 1.0 - lower_prob
#         lower_bound = beta.ppf(lower_prob, self.alpha, self.beta)
#         upper_bound = beta.ppf(upper_prob, self.alpha, self.beta)
#         return lower_bound, upper_bound
#

class OnlineBinomialModel:
    def __init__(self, alpha_prior=1.0, beta_prior=1.0, forgetting_factor=0.85):
        self.alpha = alpha_prior
        self.beta = beta_prior
        self.ff = forgetting_factor

    def update(self, successes, trials):
        failures = trials - successes
        # Apply forgetting factor to old memory, then add new data
        self.alpha = (self.alpha * self.ff) + successes
        self.beta = (self.beta * self.ff) + failures

    def get_expected_probability(self):
        return self.alpha / (self.alpha + self.beta)

    def get_credible_interval(self, confidence=0.95):
        lp = (1.0 - confidence) / 2.0
        up = 1.0 - lp
        return beta.ppf(lp, self.alpha, self.beta), beta.ppf(up, self.alpha, self.beta)
# 1. GENERATE SYNTHETIC DATA
np.random.seed(42)
n_steps = 50
trials_per_step = 10  # We test 10 users at each step

true_p_phase1 = 0.2
true_p_phase2 = 0.7

data_list = []  # Will store tuples of (successes, trials)
for i in range(n_steps):
    if i < 25:
        successes = np.random.binomial(trials_per_step, true_p_phase1)
    else:
        successes = np.random.binomial(trials_per_step, true_p_phase2)
    data_list.append((successes, trials_per_step))

# 2. INITIALIZE MODEL
model = OnlineBinomialModel(alpha_prior=1.0, beta_prior=1.0)

# Lists to store history for plotting
expected_probs = []
lower_bounds = []
upper_bounds = []
observed_ratios = []

# 3. UPDATE MODEL IN A LOOP
for successes, trials in data_list:
    model.update(successes=successes, trials=trials)

    expected_probs.append(model.get_expected_probability())
    lb, ub = model.get_credible_interval(0.95)
    lower_bounds.append(lb)
    upper_bounds.append(ub)

    # Calculate the raw observed ratio for this batch of trials
    observed_ratios.append(successes / trials)

# 4. PLOT THE RESULTS
plt.figure(figsize=(12, 6))

# Plot the raw observed data points
plt.scatter(range(1, n_steps + 1), observed_ratios, alpha=0.4, color='gray', label='Observed Batch Ratio')

# Plot the true underlying probability
plt.plot(range(1, n_steps + 1), [true_p_phase1] * 25 + [true_p_phase2] * 25,
         'k--', linewidth=2, label='True Probability')

# Plot the model's evolving expected probability
plt.plot(range(1, n_steps + 1), expected_probs, 'g-', linewidth=2, label='Model Expected Probability')

# Plot the 95% Credible Interval
plt.fill_between(range(1, n_steps + 1), lower_bounds, upper_bounds,
                 color='green', alpha=0.2, label='95% Credible Interval')

plt.title('Self-Updating Online Binomial Model', fontsize=14)
plt.xlabel('Time Step (Batches of 10 Trials)', fontsize=12)
plt.ylabel(r'Probability of Success (\(p\))', fontsize=12)
plt.ylim(-0.05, 1.05)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/binomial_graph.png', dpi=300, bbox_inches='tight')
plt.show()