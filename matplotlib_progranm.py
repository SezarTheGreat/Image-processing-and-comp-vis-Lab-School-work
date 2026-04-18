import matplotlib.pyplot as plt
import random

# Given / defined arrays
x1 = [1, 2, 3, 4, 5]
y1 = [2, 4, 6, 8, 10]

# Randomly generated arrays
x2 = random.sample(range(1, 11), 5)
y2 = random.sample(range(1, 11), 5)

# Plotting
plt.plot(x1, y1, marker='o', mfc="y", mec="red", ms=5, ls=":", label='x1 vs y1')
plt.plot(x2, y2, marker='s', mfc="y", mec="red", ms=7, ls="--", lw=10, label='x2 vs y2')

# Grid
plt.grid(linestyle="--", color="r", lw=5)

# Labels with SAME style as title
plt.xlabel("X values", color="blue", fontsize=10,loc="right")
plt.ylabel("Y values", color="blue", fontsize=10,loc="top")

# Title
plt.title("Random Title", color="blue", fontsize=10, loc="left")
plt.legend()
plt.show()