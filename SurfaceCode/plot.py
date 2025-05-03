import pickle



with open('results.pickle', 'rb') as handle:
    b = pickle.load(handle)

import matplotlib.pyplot as plt
import numpy as np
p_list = np.linspace(0.01, 1, 10)

# Create a figure with 1 row and 3 columns of subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 6))

# Plot on first subplot
ax1.plot(p_list, b["5"], color='red',label = "distance 5",marker = "*")
ax1.plot(p_list, b["5 - no correction"], color='blue',label = "distance 5 - no correction", marker="o")

ax1.legend()
ax1.grid(True)

# Plot on first subplot
ax2.plot(p_list, b["7"], color='red',label = "distance 7",marker = "*")
ax2.plot(p_list, b["7 - no correction"], color='blue',label = "distance 7 - no correction", marker="o")


ax2.legend()
ax2.grid(True)

ax3.plot(p_list, b["9"], color='red',label = "distance 9",marker = "*")
ax3.plot(p_list, b["9 - no correction"], color='blue',label = "distance 9 - no correction", marker="o")

ax3.legend()
ax3.grid(True)

# Adjust layout to prevent overlapping
# plt.tight_layout()
# Only set ylabel on first subplot
ax1.set_ylabel('Logical Error Rate')

# Set xlabel on all subplots
for ax in [ax1, ax2, ax3]:
    ax.set_xlabel('Physical Error Rate (p)')

# Show the plot

plt.suptitle("Logical Error Rate vs Physical Error Rate for Different Distances d", y=1.05, fontsize=14, fontweight='bold')

plt.show()

fig.savefig("results.pdf")