import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#
# class OnlineNormalModel:
#     def __init__(self, mu_prior=0.0, var_prior=100.0, known_var=16.0):
#         self.mu = mu_prior
#         self.var = var_prior
#         self.known_var = known_var  # The variance of the data itself
#
#     def update(self, x):
#         # Convert variances to precisions (1/variance)
#         prior_precision = 1.0 / self.var
#         data_precision = 1.0 / self.known_var
#
#         # Calculate new precision and new variance
#         new_precision = prior_precision + data_precision
#         self.var = 1.0 / new_precision
#
#         # Calculate new mean (precision-weighted average)
#         self.mu = (prior_precision * self.mu + data_precision * x) / new_precision
#
#     def get_credible_interval(self, confidence=0.95):
#         # Calculate the bounds of the normal distribution for the mean
#         lower_prob = (1.0 - confidence) / 2.0
#         upper_prob = 1.0 - lower_prob
#
#         std_dev = np.sqrt(self.var)
#         lower_bound = norm.ppf(lower_prob, loc=self.mu, scale=std_dev)
#         upper_bound = norm.ppf(upper_prob, loc=self.mu, scale=std_dev)
#         return lower_bound, upper_bound


class OnlineNormalModel:
    def __init__(self, mu_prior=0.0, var_prior=100.0, known_var=16.0, forgetting_factor=0.85):
        self.mu = mu_prior
        self.var = var_prior
        self.known_var = known_var
        self.ff = forgetting_factor

    def update(self, x):
        prior_precision = 1.0 / self.var
        # Discount the prior precision (inflate old variance)
        discounted_precision = prior_precision * self.ff
        data_precision = 1.0 / self.known_var

        # Calculate new precision and variance
        new_precision = discounted_precision + data_precision
        self.var = 1.0 / new_precision

        # Calculate new mean
        self.mu = (discounted_precision * self.mu + data_precision * x) / new_precision

    def get_credible_interval(self, confidence=0.95):
        lp = (1.0 - confidence) / 2.0
        up = 1.0 - lp
        std_dev = np.sqrt(self.var)
        return norm.ppf(lp, loc=self.mu, scale=std_dev), norm.ppf(up, loc=self.mu, scale=std_dev)

# 1. GENERATE SYNTHETIC DATA
np.random.seed(42)
n_steps = 100
true_var = 16.0  # Measurement noise (standard deviation of 4)

true_mean_phase1 = 10
true_mean_phase2 = 25

data_list = []
for i in range(n_steps):
    if i < 50:
        data_list.append(np.random.normal(true_mean_phase1, np.sqrt(true_var)))
    else:
        data_list.append(np.random.normal(true_mean_phase2, np.sqrt(true_var)))

# 2. INITIALIZE MODEL
# We start with a very weak prior (mu=0, variance=100, meaning we have no idea)
model = OnlineNormalModel(mu_prior=0.0, var_prior=100.0, known_var=true_var)

# Lists to store history for plotting
expected_means = []
lower_bounds = []
upper_bounds = []

# 3. UPDATE MODEL IN A LOOP
for x in data_list:
    model.update(x)

    expected_means.append(model.mu)
    lb, ub = model.get_credible_interval(0.95)
    lower_bounds.append(lb)
    upper_bounds.append(ub)

# 4. PLOT THE RESULTS
plt.figure(figsize=(12, 6))

# Plot the raw data points
plt.scatter(range(1, n_steps + 1), data_list, alpha=0.3, color='gray', label='Raw Observations (Noisy Data)')

# Plot the true underlying mean
plt.plot(range(1, n_steps + 1), [true_mean_phase1] * 50 + [true_mean_phase2] * 50,
         'k--', linewidth=2, label='True Mean')

# Plot the model's evolving expected mean
plt.plot(range(1, n_steps + 1), expected_means, 'r-', linewidth=2, label='Model Estimated Mean')

# Plot the 95% Credible Interval for the MEAN
plt.fill_between(range(1, n_steps + 1), lower_bounds, upper_bounds,
                 color='red', alpha=0.2, label='95% Credible Interval')

plt.title('Self-Updating Online Normal Model (Estimating the Mean)', fontsize=14)
plt.xlabel('Time Step', fontsize=12)
plt.ylabel(r'Value (\(x\))', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/normal_graph.png', dpi=300, bbox_inches='tight')
plt.show()