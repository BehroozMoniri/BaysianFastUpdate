import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


class OnlineGeometricModel:
    def __init__(self, alpha_prior=1.0, beta_prior=1.0):
        # Uniform prior (Beta 1,1)
        self.alpha = alpha_prior
        self.beta = beta_prior

    def update(self, failures):
        """
        Update the model with the number of failures observed
        before the first success.
        """
        self.alpha += 1  # 1 success occurred
        self.beta += failures  # Add the observed failures

    def get_expected_probability(self):
        return self.alpha / (self.alpha + self.beta)

    def get_credible_interval(self, confidence=0.95):
        lower_prob = (1.0 - confidence) / 2.0
        upper_prob = 1.0 - lower_prob
        lower_bound = beta.ppf(lower_prob, self.alpha, self.beta)
        upper_bound = beta.ppf(upper_prob, self.alpha, self.beta)
        return lower_bound, upper_bound


# 1. GENERATE SYNTHETIC DATA
np.random.seed(42)
n_experiments = 50

# True probability of success changes halfway through
true_p_phase1 = 0.3
true_p_phase2 = 0.8

data_list = []
for i in range(n_experiments):
    # np.random.geometric(p) returns the number of trials (k >= 1)
    # To get failures before first success, we do: trials - 1
    if i < 25:
        trials = np.random.geometric(true_p_phase1)
    else:
        trials = np.random.geometric(true_p_phase2)

    failures_before_success = trials - 1
    data_list.append(failures_before_success)

# 2. INITIALIZE MODEL
model = OnlineGeometricModel(alpha_prior=1.0, beta_prior=1.0)

# Lists for plotting
expected_probs = []
lower_bounds = []
upper_bounds = []

# 3. UPDATE MODEL IN A LOOP
for failures in data_list:
    model.update(failures=failures)

    expected_probs.append(model.get_expected_probability())
    lb, ub = model.get_credible_interval(0.95)
    lower_bounds.append(lb)
    upper_bounds.append(ub)

# 4. PLOT THE RESULTS
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 2]})

# Top plot: The raw data (Failures before success)
ax1.bar(range(1, n_experiments + 1), data_list, color='gray', alpha=0.6)
ax1.set_title('Raw Observations: Number of Failures Before Success', fontsize=12)
ax1.set_ylabel('Failures (x)', fontsize=10)
ax1.set_xlabel('Experiment Number', fontsize=10)
ax1.grid(True, alpha=0.3)

# Bottom plot: The evolving probability
ax2.plot(range(1, n_experiments + 1), [true_p_phase1] * 25 + [true_p_phase2] * 25,
         'k--', linewidth=2, label='True Probability of Success')

ax2.plot(range(1, n_experiments + 1), expected_probs, 'purple', linewidth=2, label='Model Estimated Probability')
ax2.fill_between(range(1, n_experiments + 1), lower_bounds, upper_bounds,
                 color='purple', alpha=0.2, label='95% Credible Interval')

ax2.set_title('Self-Updating Online Geometric Model', fontsize=14)
ax2.set_xlabel('Experiment Number', fontsize=12)
ax2.set_ylabel(r'Probability of Success (\(p\))', fontsize=12)
ax2.set_ylim(-0.05, 1.05)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


class OnlineGeometricModel:
    def __init__(self, alpha_prior=1.0, beta_prior=1.0, forgetting_factor=0.85):
        self.alpha = alpha_prior
        self.beta = beta_prior
        self.forgetting_factor = forgetting_factor

    def update(self, failures):
        # Apply the forgetting factor to shrink old memory, then add new data
        self.alpha = (self.alpha * self.forgetting_factor) + 1
        self.beta = (self.beta * self.forgetting_factor) + failures

    def get_expected_probability(self):
        return self.alpha / (self.alpha + self.beta)

    def get_credible_interval(self, confidence=0.95):
        lower_prob = (1.0 - confidence) / 2.0
        upper_prob = 1.0 - lower_prob
        lower_bound = beta.ppf(lower_prob, self.alpha, self.beta)
        upper_bound = beta.ppf(upper_prob, self.alpha, self.beta)
        return lower_bound, upper_bound


# 1. GENERATE SYNTHETIC DATA
np.random.seed(42)
n_experiments = 50
true_p_phase1 = 0.3
true_p_phase2 = 0.8

data_list = []
for i in range(n_experiments):
    if i < 25:
        trials = np.random.geometric(true_p_phase1)
    else:
        trials = np.random.geometric(true_p_phase2)
    data_list.append(trials - 1)

# 2. INITIALIZE MODEL
# Try changing the forgetting factor (e.g., 1.0 is no forgetting, 0.8 is fast forgetting)
model = OnlineGeometricModel(alpha_prior=1.0, beta_prior=1.0, forgetting_factor=0.85)

# Lists for plotting
expected_probs = []
lower_bounds = []
upper_bounds = []

# 3. UPDATE MODEL IN A LOOP
for failures in data_list:
    model.update(failures=failures)
    expected_probs.append(model.get_expected_probability())
    lb, ub = model.get_credible_interval(0.95)
    lower_bounds.append(lb)
    upper_bounds.append(ub)

# 4. PLOT THE RESULTS
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 2]})

ax1.bar(range(1, n_experiments + 1), data_list, color='gray', alpha=0.6)
ax1.set_title('Raw Observations: Failures Before Success', fontsize=12)
ax1.set_ylabel('Failures (x)', fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, n_experiments + 1), [true_p_phase1] * 25 + [true_p_phase2] * 25,
         'k--', linewidth=2, label='True Probability of Success')

ax2.plot(range(1, n_experiments + 1), expected_probs, 'purple', linewidth=2, label='Model Estimated Probability')
ax2.fill_between(range(1, n_experiments + 1), lower_bounds, upper_bounds,
                 color='purple', alpha=0.2, label='95% Credible Interval')

ax2.set_title('Self-Updating Geometric Model (With Exponential Forgetting)', fontsize=14)
ax2.set_xlabel('Experiment Number', fontsize=12)
ax2.set_ylabel(r'Probability of Success (\(p\))', fontsize=12)
ax2.set_ylim(-0.05, 1.05)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/geometric_graph.png', dpi=300, bbox_inches='tight')
plt.show()