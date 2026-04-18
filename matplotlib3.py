import numpy as np
import matplotlib.pyplot as plt

x = np.array(["A", "B", "C", "D","E","F","G","H","I","J"])
y = np.array([25, 40, 30, 35,50,45,60,98,90,67])

fig, ax_pie = plt.subplots(1, 1, figsize=(6, 4))

colors = plt.cm.tab10.colors
ax_pie.pie(
    y,
    labels=x,
    colors=colors,
    autopct="%1.1f%%",
    startangle=90,
)
ax_pie.set_title("Pie Chart")

plt.tight_layout()
plt.show()
