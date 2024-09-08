"""
NOTE: All of the imports will be at the top, but the
functions will be defined in each "section" for educational purposes.
In practice, all functions should be at the top
"""
import numpy
import matplotlib.pyplot as plt
import pathlib

DIR_OUT = pathlib.Path(__file__).resolve().parents[1] / "out" / "micrograd"
DIR_OUT.mkdir(exist_ok=True, parents=True)

# Derivative of a single variable =================================================================

def f(x):
    """
    3x^2 - 4x + 5
    """
    return 3*x**2 - 4*x + 5

def f_prime(x):
    """
    f'(x)
    """
    return 6*x - 4

xs = numpy.arange(-6,6,2e-05)
ys = f(xs)
yss = f_prime(xs) # exact

hs = numpy.arange(0.00001,6.00001,0.00001)
ss = (f(xs + hs) - f(xs)) / hs # approximation of rise/run

plt.plot(xs, ys)
plt.plot(xs, yss)
plt.plot(xs, ss)
plt.savefig(DIR_OUT / "f1.png")

# Derivative of many variables ====================================================================

def F(x1, x2, x3):
    return x1*x2 + x3

x = [3, -4, 10]

y = F(*x) # F(x1, x2, x3)
h = 0.00001
"""
Take x_i += h to get x_h = [x1, x2, ..., xi + h, ..., xn]
Evaluate (F(x_h) - F(x)) / h to get the derivative for that x_i
"""
print("F(x)", "y_h", "s")
for i in range(len(x)):
    x_p = x[::] # copy
    x_p[i] += h # increase xi by a small amount
    y_h = F(*x_p)
    y = F(*x)
    s = (y_h-y)/h
    print(y, y_h, s)