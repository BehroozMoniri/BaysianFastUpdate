import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma


# class FastOnlineExponentialModel:
#     def __init__(self, alpha_prior=1.0, beta_prior=1.0, surprise_threshold=3.0):
#         self.alpha_0 = alpha_prior
#         self.beta_0 = beta_prior
#         self.alpha = alpha_prior
#         self.beta = beta_prior
#         self.threshold = surprise_threshold
#
#     def update(self, wait_time):
#         # 1. Calculate current expected rate (lambda) and standard deviation
#         expected_rate = self.alpha / self.beta
#         std_dev = np.sqrt(self.alpha) / self.beta
#
#         # 2. Check for Surprise (Change Point Detection)
#         # Note: For continuous time, we compare the observed wait time's implied rate
#         # to the expected rate, but we do it directly on the rate scale.
#         implied_rate = 1.0 / wait_time if wait_time > 0 else 1000.0
#
#         if abs(implied_rate - expected_rate) > self.threshold * std_dev:
#             # SURPRISE! Reset memory to learn the new rate immediately
#             self.alpha = self.alpha_0
#             self.beta = self.beta_0
#
#         # 3. Standard Bayesian Update for Exponential (Gamma conjugate)
#         self.alpha += 1  # We observed 1 event
#         self.beta += wait_time  # We waited this much time
#
#     def get_expected_rate(self):
#         return self.alpha / self.beta
#
#     def get_expected_wait_time(self):
#         return 1.0 / self.get_expected_rate()
#
#     def get_credible_interval(self, confidence=0.95):
#         lp = (1.0 - confidence) / 2.0
#         up = 1.0 - lp
#         scale = 1.0 / self.beta
#         return gamma.ppf(lp, a=self.alpha, scale=scale), gamma.ppf(up, a=self.alpha, scale=scale)
#
#
# # 1. GENERATE DATA
# np.random.seed(42)
# n_steps = 50
# true_rate_phase1 = 2.0  # Fast arrivals (0.5s wait)
# true_rate_phase2 = 0.25  # Slow arrivals (4.0s wait)
#
# data_list = []
# for i in range(n_steps):
#     if i < 25:
#         wait_time = np.random.exponential(1.0 / true_rate_phase1)
#     else:
#         wait_time = np.random.exponential(1.0 / true_rate_phase2)
#     data_list.append(wait_time)
#
# # 2. INITIALIZE MODELS
# standard_model = FastOnlineExponentialModel(surprise_threshold=99999)  # No surprise reset
# fast_model = FastOnlineExponentialModel(surprise_threshold=2.5)  # Sensitive reset
#
# # Lists for plotting
# std_rates, fast_rates = [], []
# fast_lb, fast_ub = [], []
#
# # 3. RUN SIMULATION
# for wait_time in data_list:
#     standard_model.update(wait_time)
#     fast_model.update(wait_time)
#
#     std_rates.append(standard_model.get_expected_wait_time())
#     fast_rates.append(fast_model.get_expected_wait_time())
#
#     lb, ub = fast_model.get_credible_interval()
#     # Invert CI bounds because we are plotting wait time (1/rate)
#     fast_lb.append(1.0 / ub if ub > 0 else 0)
#     fast_ub.append(1.0 / lb if lb > 0 else 10)
#
# # 4. PLOT
# plt.figure(figsize=(12, 6))
#
# # Plot the raw waiting times
# plt.scatter(range(1, n_steps + 1), data_list, alpha=0.4, color='gray', label='Observed Wait Times')
# plt.plot(range(1, n_steps + 1), [1.0 / true_rate_phase1] * 25 + [1.0 / true_rate_phase2] * 25,
#          'k--', linewidth=2, label='True Wait Time')
#
# plt.plot(range(1, n_steps + 1), std_rates, 'b-', linewidth=2, alpha=0.5, label='Standard Bayes (Slow)')
# plt.plot(range(1, n_steps + 1), fast_rates, 'r-', linewidth=2.5, label='Surprise Reset Model (Instant)')
# plt.fill_between(range(1, n_steps + 1), fast_lb, fast_ub, color='red', alpha=0.15, label='Fast Model 95% CI')
#
# plt.title('Self-Updating Exponential Model (Tracking Wait Times)', fontsize=14)
# plt.xlabel('Event Number', fontsize=12)
# plt.ylabel('Wait Time (Seconds)', fontsize=12)
# plt.ylim(0, 8)  # Cap y-axis for better visibility
# plt.legend(loc='upper left')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma


class FastOnlineExponentialModel:
    def __init__(self, alpha_prior=1.0, beta_prior=1.0, forgetting_factor=0.90):
        self.alpha_0 = alpha_prior
        self.beta_0 = beta_prior
        self.alpha = alpha_prior
        self.beta = beta_prior
        self.ff = forgetting_factor

    def update(self, wait_time):
        # Apply forgetting factor to old memory, then add new data
        self.alpha = (self.alpha * self.ff) + 1
        self.beta = (self.beta * self.ff) + wait_time

    def get_expected_wait_time(self):
        # The expected rate is alpha/beta. Wait time is the inverse.
        return self.beta / self.alpha

    def get_credible_interval(self, confidence=0.95):
        lp = (1.0 - confidence) / 2.0
        up = 1.0 - lp
        scale = 1.0 / self.beta

        # Get bounds for the rate (lambda)
        lb_rate = gamma.ppf(lp, a=self.alpha, scale=scale)
        ub_rate = gamma.ppf(up, a=self.alpha, scale=scale)

        # Invert bounds to get wait time (1/rate)
        # Note: we swap ub and lb because 1/x flips the order
        return 1.0 / ub_rate, 1.0 / lb_rate


# 1. GENERATE DATA
np.random.seed(42)
n_steps = 100
true_rate_phase1 = 2.0  # Fast arrivals (0.5s wait)
true_rate_phase2 = 0.25  # Slow arrivals (4.0s wait)

data_list = []
for i in range(n_steps):
    if i < int(n_steps /2):
        wait_time = np.random.exponential(1.0 / true_rate_phase1)
    else:
        wait_time = np.random.exponential(1.0 / true_rate_phase2)
    data_list.append(wait_time)

# 2. INITIALIZE MODELS
# Standard Bayes (No forgetting: ff=1.0)
standard_model = FastOnlineExponentialModel(forgetting_factor=1.0)
# Fast Model (Heavy forgetting: only remembers ~10 recent events)
fast_model = FastOnlineExponentialModel(forgetting_factor=0.9)

# Lists for plotting
std_wait_times, fast_wait_times = [], []
fast_lb, fast_ub = [], []

# 3. RUN SIMULATION
for wait_time in data_list:
    standard_model.update(wait_time)
    fast_model.update(wait_time)

    std_wait_times.append(standard_model.get_expected_wait_time())
    fast_wait_times.append(fast_model.get_expected_wait_time())

    lb, ub = fast_model.get_credible_interval()
    fast_lb.append(lb)
    fast_ub.append(ub)

# 4. PLOT
plt.figure(figsize=(12, 6))

# Plot the raw waiting times
plt.scatter(range(1, n_steps + 1), data_list, alpha=0.4, color='gray', label='Observed Wait Times')
plt.plot(range(1, n_steps + 1), [1.0 / true_rate_phase1] * int(n_steps/2) + [1.0 / true_rate_phase2] * int(n_steps/2),
         'k--', linewidth=2, label='True Wait Time')

plt.plot(range(1, n_steps + 1), std_wait_times, 'b-', linewidth=2, alpha=0.5, label='Standard Bayes (Slow)')
plt.plot(range(1, n_steps + 1), fast_wait_times, 'r-', linewidth=2.5, label='Forgetting Model (Fast)')
plt.fill_between(range(1, n_steps + 1), fast_lb, fast_ub, color='red', alpha=0.15, label='Fast Model 95% CI')

plt.title('Self-Updating Exponential Model (With Forgetting Factor)', fontsize=14)
plt.xlabel('Event Number', fontsize=12)
plt.ylabel('Wait Time (Seconds)', fontsize=12)
plt.ylim(0, 8)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/exponential_graph.png', dpi=300, bbox_inches='tight')
plt.show()