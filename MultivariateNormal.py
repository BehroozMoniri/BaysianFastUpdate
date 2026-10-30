import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure the images directory exists
os.makedirs('images', exist_ok=True)


class FastOnlineDirichletModel:
    def __init__(self, n_categories, alpha_prior=1.0, forgetting_factor=1.0):
        self.alphas = np.ones(n_categories) * alpha_prior
        self.ff = forgetting_factor

    def update(self, counts_vector):
        # Apply forgetting factor to the entire alpha vector, then add new counts
        self.alphas = (self.alphas * self.ff) + counts_vector

    def get_expected_probabilities(self):
        return self.alphas / np.sum(self.alphas)


# 1. GENERATE DATA
np.random.seed(42)
n_steps = 100
n_categories = 3
trials_per_step = 50  # INCREASED: 50 trials makes the data less noisy

true_probs_phase1 = np.array([0.2, 0.5, 0.3])  # Red, Green, Blue
true_probs_phase2 = np.array([0.7, 0.2, 0.1])  # Red, Green, Blue

data_list = []
for i in range(n_steps):
    true_probs = true_probs_phase1 if i < 50 else true_probs_phase2
    counts = np.random.multinomial(trials_per_step, true_probs)
    data_list.append(counts)

# 2. INITIALIZE MODELS
# We will use 0.90 for the fast model.
std_model = FastOnlineDirichletModel(n_categories, forgetting_factor=1.0)
fast_model = FastOnlineDirichletModel(n_categories, forgetting_factor=0.90)

# Lists for plotting
std_probs_history = []
fast_probs_history = []

# 3. RUN SIMULATION
for counts in data_list:
    std_model.update(counts)
    fast_model.update(counts)

    std_probs_history.append(std_model.get_expected_probabilities())
    fast_probs_history.append(fast_model.get_expected_probabilities())

std_probs_history = np.array(std_probs_history)
fast_probs_history = np.array(fast_probs_history)

# Create the "True" probability arrays for plotting
true_history = np.array([true_probs_phase1] * 50 + [true_probs_phase2] * 50)

# 4. PLOT
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
colors = ['red', 'green', 'blue']
cat_names = ['Category 1 (Red)', 'Category 2 (Green)', 'Category 3 (Blue)']

for i in range(3):
    # Top Plot: Standard Model
    ax1.plot(range(1, n_steps + 1), true_history[:, i], color=colors[i], linestyle='--', linewidth=2, alpha=0.7,
             label=f'True {cat_names[i]}')
    ax1.plot(range(1, n_steps + 1), std_probs_history[:, i], color=colors[i], linewidth=2, label=f'Est. {cat_names[i]}')

    # Bottom Plot: Fast Model
    ax2.plot(range(1, n_steps + 1), true_history[:, i], color=colors[i], linestyle='--', linewidth=2, alpha=0.7,
             label=f'True {cat_names[i]}')
    ax2.plot(range(1, n_steps + 1), fast_probs_history[:, i], color=colors[i], linewidth=2,
             label=f'Est. {cat_names[i]}')

# Formatting Top Plot
ax1.axvline(x=50, color='black', linestyle=':', linewidth=2, label='Distribution Shift')
ax1.set_title('Standard Dirichlet Model (Slow to Adapt)', fontsize=14)
ax1.set_ylabel('Probability', fontsize=12)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Formatting Bottom Plot
ax2.axvline(x=50, color='black', linestyle=':', linewidth=2, label='Distribution Shift')
ax2.set_title('Fast Dirichlet Model (Forgetting Factor = 0.90)', fontsize=14)
ax2.set_xlabel('Time Step (Batches of 50 Trials)', fontsize=12)
ax2.set_ylabel('Probability', fontsize=12)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save to the images folder
plt.savefig('images/dirichlet_graph.png', dpi=300, bbox_inches='tight')
plt.show()