import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([1,2,3,4,5,6,7,8,9,10])
y1 = np.array([2,4,1,5,7,3,8,6,9,10])

x2 = np.array([1,2,3,4,5,6,7,8,9,10])
y2 = np.array([1,4,9,16,25,36,49,64,81,100])

x3 = np.array([1,2,3,4,5,6,7,8,9,10])
y3 = np.array([1,1.4,1.7,2,2.2,2.4,2.6,2.8,3,3.2])

x4 = np.array([1,2,3,4,5,6,7,8,9,10])
y4 = np.array([0,0.7,1.1,1.4,1.6,1.8,1.9,2.1,2.2,2.3])

sizes  = [50,80,110,140,170,200,230,260,290,320]
colors = ['red','blue','green','purple','orange',
          'brown','pink','gray','cyan','black']

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0,0].scatter(x1, y1, s=sizes, c=colors,marker="*")
axs[0,0].set_title("Plot 1")
axs[0,0].set_xlabel("X")
axs[0,0].set_ylabel("Y")
axs[0,0].grid(color="lightgray", linestyle="-.")

axs[0,1].scatter(x2, y2, s=sizes, c=colors)
axs[0,1].set_title("Plot 2")
axs[0,1].set_xlabel("X")
axs[0,1].set_ylabel("Y")
axs[0,1].grid(color="lightblue", linestyle="-.")

axs[1,0].scatter(x3, y3, s=sizes, c=colors,marker="^")
axs[1,0].set_title("Plot 3")
axs[1,0].set_xlabel("X")
axs[1,0].set_ylabel("Y")
axs[1,0].grid(color="lightgreen", linestyle="-.")

axs[1,1].scatter(x4, y4, s=sizes, c=colors,marker="+")
axs[1,1].set_title("Plot 4")
axs[1,1].set_xlabel("X")
axs[1,1].set_ylabel("Y")
axs[1,1].grid(color="mistyrose", linestyle="-.")

plt.tight_layout()
plt.show()
