import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3 * x**2 - 4 * x + 5

xs = np.arange(-5, 5, 0.25)  # Fixed typo here
ys = f(xs)

plt.plot(xs, ys)
plt.title("Plot of f(x) = 3x^2 - 4x + 5")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()


