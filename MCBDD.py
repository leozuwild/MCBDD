import numpy as np
import matplotlib.pyplot as plt

# Setup
prevalence_range = np.linspace(0.00001, 0.5, 100)
specificity_values = [0.99, 0.999, 0.9999, 0.99999]
sensitivity = 0.99
probability_matrix = np.zeros((len(prevalence_range), len(specificity_values)))

# Calculations
for i, prevalence in enumerate(prevalence_range):
    for j, specificity in enumerate(specificity_values):
        false_positive_rate = 1 - specificity
        P_positive = (sensitivity * prevalence) + (false_positive_rate * (1 - prevalence))
        if P_positive == 0:
            probability_matrix[i, j] = np.nan
        else:
            probability_matrix[i, j] = (sensitivity * prevalence) / P_positive

# Plot results
plt.figure(figsize=(8, 5))
for j, specificity in enumerate(specificity_values):
    plt.plot(prevalence_range * 100, probability_matrix[:, j] * 100, label=f"Specificity {specificity*100:.3f}%")

# Customize plot
plt.xlabel('Prevalence (%)')
plt.ylabel('P(Infected | Positive Test) (%)')
plt.title('Probability of True Infection vs. Prevalence & Specificity')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
