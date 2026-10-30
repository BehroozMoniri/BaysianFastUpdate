class OnlinePoissonModel:
    def __init__(self, alpha_prior=1.0, beta_prior=1.0):
        # Initialize with Gamma prior parameters
        self.alpha = alpha_prior
        self.beta = beta_prior

    def update(self, total_events, time_intervals):
        """
        Update the posterior with new data.
        - total_events: sum of counts observed (e.g., 15 cars)
        - time_intervals: number of intervals (e.g., 3 hours)
        """
        # Update rules derived from Bayes' Theorem
        self.alpha = self.alpha + total_events
        self.beta = self.beta + time_intervals

    def get_expected_rate(self):
        # The mean of a Gamma(alpha, beta) distribution is alpha / beta
        return self.alpha / self.beta


# --- Example Usage ---
model = OnlinePoissonModel(alpha_prior=1.0, beta_prior=1.0)

# Day 1: Observe 12 events in 4 hours
model.update(total_events=12, time_intervals=4)
print(f"Day 1 Expected Rate: {model.get_expected_rate():.2f} events/hour")
# (1+12)/(1+4) = 13/5 = 2.6

# Day 2: Observe 8 events in 2 hours
model.update(total_events=8, time_intervals=2)
print(f"Day 2 Expected Rate: {model.get_expected_rate():.2f} events/hour")
# (13+8)/(5+2) = 21/7 = 3.0

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma


# class OnlinePoissonModel:
#     def __init__(self, alpha_prior=1.0, beta_prior=1.0):
#         self.alpha = alpha_prior
#         self.beta = beta_prior
#
#     def update(self, observed_count, time_interval=1.0):
#         # The core Bayesian update
#         self.alpha += observed_count
#         self.beta += time_interval
#
#     def get_expected_rate(self):
#         return self.alpha / self.beta
#
#     def get_credible_interval(self, confidence=0.95):
#         # Calculate the lower and upper bounds of the Gamma distribution
#         lower_prob = (1.0 - confidence) / 2.0
#         upper_prob = 1.0 - lower_prob
#
#         # Note: scipy's gamma uses 'scale' which is 1/rate (1/beta)
#         scale = 1.0 / self.beta
#         lower_bound = gamma.ppf(lower_prob, a=self.alpha, scale=scale)
#         upper_bound = gamma.ppf(upper_prob, a=self.alpha, scale=scale)
#         return lower_bound, upper_bound

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma


# class FastOnlinePoissonModel:
#     def __init__(self, alpha_prior=1.0, beta_prior=1.0, initial_lr=0.5, decay_rate=0.1):
#         self.alpha = alpha_prior
#         self.beta = beta_prior
#         self.alpha_0 = alpha_prior
#         self.beta_0 = beta_prior
#         self.initial_lr = initial_lr
#         self.decay_rate = decay_rate
#         self.t = 0
#
#     def update(self, observed_count, time_interval=1.0):
#         self.t += 1
#         # Learning rate starts at 'initial_lr' and decays by 1/t
#         # We add a small constant to prevent division by zero at t=1
#         lr = max(self.initial_lr / (1 + self.decay_rate * self.t), 0.01)
#
#         # Dynamic Update Rule
#         self.alpha = (1 - lr) * self.alpha + lr * (observed_count + self.alpha_0)
#         self.beta = (1 - lr) * self.beta + lr * (time_interval + self.beta_0)
#
#     def get_expected_rate(self):
#         return self.alpha / self.beta
#
#     def get_credible_interval(self, confidence=0.95):
#         lp = (1.0 - confidence) / 2.0
#         up = 1.0 - lp
#         scale = 1.0 / self.beta
#         return gamma.ppf(lp, a=self.alpha, scale=scale), gamma.ppf(up, a=self.alpha, scale=scale)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma


class FastOnlinePoissonModel:
    def __init__(self, alpha_prior=1.0, beta_prior=1.0, surprise_threshold=3.0):
        self.alpha_0 = alpha_prior
        self.beta_0 = beta_prior
        self.alpha = alpha_prior
        self.beta = beta_prior
        self.threshold = surprise_threshold

    def update(self, observed_count, time_interval=1.0):
        # 1. Calculate the expected rate and standard deviation
        expected_rate = self.alpha / self.beta
        std_dev = np.sqrt(self.alpha) / self.beta

        # 2. Check for Surprise (Change Point Detection)
        if abs(observed_count - expected_rate) > self.threshold * std_dev:
            # SURPRISE! The world has changed. Reset the memory to the prior.
            self.alpha = self.alpha_0
            self.beta = self.beta_0

        # 3. Standard Bayesian Update
        self.alpha += observed_count
        self.beta += time_interval

    def get_expected_rate(self):
        return self.alpha / self.beta

    def get_credible_interval(self, confidence=0.95):
        lp = (1.0 - confidence) / 2.0
        up = 1.0 - lp
        scale = 1.0 / self.beta
        return gamma.ppf(lp, a=self.alpha, scale=scale), gamma.ppf(up, a=self.alpha, scale=scale)


# 1. GENERATE DATA (True rate 0.5 jumps to 3.0 at step 30)
np.random.seed(62)
n_steps = 60
true_rate_phase1 = 0.5
true_rate_phase2 = 3.0

data_list = []
for i in range(n_steps):
    if i < 30:
        data_list.append(np.random.poisson(true_rate_phase1))
    else:
        data_list.append(np.random.poisson(true_rate_phase2))

# 2. INITIALIZE MODELS
# Standard Bayes (No forgetting, no surprise)
standard_model = FastOnlinePoissonModel(surprise_threshold=99999)
# Fast model (Resets memory if data is >2 std devs away from expected)
fast_model = FastOnlinePoissonModel(surprise_threshold=2.0)

# Lists for plotting
std_rates, fast_rates = [], []
fast_lb, fast_ub = [], []

# 3. RUN SIMULATION
for count in data_list:
    standard_model.update(count)
    fast_model.update(count)

    std_rates.append(standard_model.get_expected_rate())
    fast_rates.append(fast_model.get_expected_rate())
    lb, ub = fast_model.get_credible_interval()
    fast_lb.append(lb);
    fast_ub.append(ub)

# 4. PLOT
plt.figure(figsize=(12, 6))
plt.bar(range(1, n_steps + 1), data_list, alpha=0.2, color='gray', label='Observed Counts')
plt.plot(range(1, n_steps + 1), [true_rate_phase1] * 30 + [true_rate_phase2] * 30, 'k--', linewidth=2,
         label='True Rate')

plt.plot(range(1, n_steps + 1), std_rates, 'b-', linewidth=2, alpha=0.5, label='Standard Bayes (Slow)')
plt.plot(range(1, n_steps + 1), fast_rates, 'r-', linewidth=2.5, label='Surprise Reset Model (Instant)')
plt.fill_between(range(1, n_steps + 1), fast_lb, fast_ub, color='red', alpha=0.15, label='Fast Model 95% CI')

plt.title('Instant Adaptation via Change Point Detection (Surprise)', fontsize=14)
plt.xlabel('Time Step')
plt.ylabel(r'Event Rate (\(\lambda\))')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(r'images\poisson_graph.png', dpi=300, bbox_inches='tight')

plt.show()


# 1. GENERATE DATA (True rate starts at 0.5, jumps to 3.0 at step 30)
# np.random.seed(42)
# n_steps = 60
# true_rate_phase1 = 0.5
# true_rate_phase2 = 3.0
#
# data_list = []
# for i in range(n_steps):
#     if i < 30:
#         data_list.append(np.random.poisson(true_rate_phase1))
#     else:
#         data_list.append(np.random.poisson(true_rate_phase2))
#
# # 2. INITIALIZE MODELS
# standard_model = FastOnlinePoissonModel(initial_lr=0.0)  # initial_lr=0 means pure Bayes
# fast_model = FastOnlinePoissonModel(initial_lr=1.0, decay_rate=0.2)
#
# # Lists for plotting
# std_rates, fast_rates = [], []
# fast_lb, fast_ub = [], []
#
# # 3. RUN SIMULATION
# for count in data_list:
#     standard_model.update(count)
#     fast_model.update(count)
#
#     std_rates.append(standard_model.get_expected_rate())
#     fast_rates.append(fast_model.get_expected_rate())
#     lb, ub = fast_model.get_credible_interval()
#     fast_lb.append(lb);
#     fast_ub.append(ub)
#
# # 4. PLOT
# plt.figure(figsize=(12, 6))
# plt.bar(range(1, n_steps + 1), data_list, alpha=0.2, color='gray', label='Observed Counts')
# plt.plot(range(1, n_steps + 1), [true_rate_phase1] * 30 + [true_rate_phase2] * 30, 'k--', linewidth=2,
#          label='True Rate')
#
# plt.plot(range(1, n_steps + 1), std_rates, 'b-', linewidth=2, alpha=0.5, label='Standard Bayes (Slow)')
# plt.plot(range(1, n_steps + 1), fast_rates, 'r-', linewidth=2.5, label='Dynamic Learning Rate (Fast)')
# plt.fill_between(range(1, n_steps + 1), fast_lb, fast_ub, color='red', alpha=0.15, label='Fast Model 95% CI')
#
# plt.title('Speeding up Poisson-Gamma Learning at the Beginning', fontsize=14)
# plt.xlabel('Time Step')
# plt.ylabel(r'Event Rate (\(\lambda\))')
# plt.legend(loc='upper left')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()
#
# # 1. GENERATE SYNTHETIC DATA
# np.random.seed(42)  # For reproducible results
# # The true rate starts at 3, then jumps to 6 halfway through
# true_rate_phase1 = 3
# true_rate_phase2 = 6
# n_steps = 100
#
# data_list = []
# for i in range(n_steps):
#     if i < 50:
#         data_list.append(np.random.poisson(true_rate_phase1))
#     else:
#         data_list.append(np.random.poisson(true_rate_phase2))
#
# # 2. INITIALIZE MODEL
# # We start with a weak prior: alpha=1, beta=1 (expected rate = 1)
# model = OnlinePoissonModel(alpha_prior=1.0, beta_prior=1.0)
#
# # Lists to store history for plotting
# expected_rates = []
# lower_bounds = []
# upper_bounds = []
#
# # 3. UPDATE MODEL IN A LOOP
# for count in data_list:
#     model.update(observed_count=count, time_interval=1.0)
#
#     # Record the current state of the model
#     expected_rates.append(model.get_expected_rate())
#     lb, ub = model.get_credible_interval(0.95)
#     lower_bounds.append(lb)
#     upper_bounds.append(ub)
#
# # 4. PLOT THE RESULTS
# plt.figure(figsize=(12, 6))
#
# # Plot the raw data as a bar chart (shifted slightly so it aligns with the steps)
# plt.bar(range(1, n_steps + 1), data_list, alpha=0.3, color='gray', label='Observed Counts')
#
# # Plot the true underlying rate
# plt.plot(range(1, n_steps + 1), [true_rate_phase1] * 50 + [true_rate_phase2] * 50,
#          'k--', linewidth=2, label='True Underlying Rate')
#
# # Plot the model's evolving expected rate
# plt.plot(range(1, n_steps + 1), expected_rates, 'b-', linewidth=2, label='Model Expected Rate')
#
# # Plot the 95% Credible Interval
# plt.fill_between(range(1, n_steps + 1), lower_bounds, upper_bounds,
#                  color='blue', alpha=0.2, label='95% Credible Interval')
#
# plt.title('Self-Updating Online Poisson Model', fontsize=14)
# plt.xlabel('Time Step', fontsize=12)
# plt.ylabel(r'Event Rate (\(\lambda\))', fontsize=12)
# plt.legend(loc='upper left')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()