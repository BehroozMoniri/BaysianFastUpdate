import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import os

# Ensure the images directory exists
os.makedirs('images', exist_ok=True)


class FastOnlineBinomialModel:
    def __init__(self, alpha_prior=1.0, beta_prior=1.0, forgetting_factor=1.0):
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


# 1. GENERATE DATA (True prob jumps from 0.2 to 0.7 at step 50)
np.random.seed(42)
n_steps = 100
trials_per_step = 10

data_list = []
for i in range(n_steps):
    true_p = 0.2 if i < 50 else 0.7
    data_list.append(np.random.binomial(trials_per_step, true_p))

# 2. INITIALIZE MODELS
# Standard Model (No forgetting: ff=1.0)
standard_model = FastOnlineBinomialModel(forgetting_factor=1.0)
# Fast Model (Forgetting: ff=0.85)
fast_model = FastOnlineBinomialModel(forgetting_factor=0.85)

# Lists for plotting
std_probs, fast_probs = [], []
fast_lb, fast_ub = [], []

# 3. RUN SIMULATION
for successes in data_list:
    standard_model.update(successes, trials_per_step)
    fast_model.update(successes, trials_per_step)

    std_probs.append(standard_model.get_expected_probability())
    fast_probs.append(fast_model.get_expected_probability())
    lb, ub = fast_model.get_credible_interval()
    fast_lb.append(lb);
    fast_ub.append(ub)

# 4. PLOT
plt.figure(figsize=(12, 6))

# Plot the true probability
plt.plot(range(1, n_steps + 1), [0.2] * 50 + [0.7] * 50, 'k--', linewidth=2, label='True Probability')

# Plot both models
plt.plot(range(1, n_steps + 1), std_probs, 'b-', linewidth=2, alpha=0.6, label='Standard Bayes (Slow)')
plt.plot(range(1, n_steps + 1), fast_probs, 'g-', linewidth=2.5, label='Forgetting Model (Fast)')

# Plot the confidence interval for the fast model
plt.fill_between(range(1, n_steps + 1), fast_lb, fast_ub, color='green', alpha=0.15, label='Fast Model 95% CI')

plt.title('Speeding up Binomial-Beta Learning via Forgetting Factor', fontsize=14)
plt.xlabel('Time Step (Batches of 10 Trials)')
plt.ylabel(r'Probability of Success (\(p\))')
plt.ylim(-0.05, 1.05)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save to the images folder
plt.savefig('images/binomial_graph.png', dpi=300, bbox_inches='tight')
plt.show()