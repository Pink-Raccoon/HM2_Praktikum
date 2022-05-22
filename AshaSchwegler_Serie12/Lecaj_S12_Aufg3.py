import numpy as np
import matplotlib.pyplot as plt


def euler(f, x, h, y0):
    y = np.full(x.shape[0], 0, dtype=np.float64)
    y[0] = y0

    for i in range(x.size - 1):
        y[i + 1] = y[i] + h * f(x[i], y[i])

    return y


def mittelpunkt(f, x, h, y0):
    y = np.full(x.shape[0], 0, dtype=np.float64)
    y[0] = y0

    for i in range(x.size - 1):
        y[i + 1] = y[i] + h * f(x[i] + (h / 2.0), y[i] + (h / 2.0) * f(x[i], y[i]))

    return y


def mod_euler(f, x, h, y0):
    y = np.full(x.shape[0], 0, dtype=np.float64)
    y[0] = y0

    for i in range(x.size - 1):
        y[i + 1] = y[i] + h * (f(x[i], y[i]) + f(x[i + 1], y[i] + h * f(x[i], y[i]))) / 2

    return y

def runge_kutta(f, x, h, y0):
    y = np.full(x.shape[0], 0, dtype=np.float64)
    y[0] = y0

    for i in range(x.size - 1):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + (h / 2.0), y[i] + (h / 2.0) * k1)
        k3 = f(x[i] + (h / 2.0), y[i] + (h / 2.0) * k2)
        k4 = f(x[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h * (1 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return y


def f(x, y):
    return (x ** 2)/ y

def f_exact(x):
    return np.sqrt(((2*x**3)/3)+4)

a = 0
b = 10
y0 = 2
h = 0.1

x = np.arange(a, b + h, h, dtype=np.float64)

y_euler = euler(f, x, h, y0)
y_mittelpunkt = mittelpunkt(f, x, h, y0)
y_mod_euler = mod_euler(f, x, h, y0)
y_runge_kutta = runge_kutta(f, x, h, y0)

y_exact = f_exact(x)

print('x = ' + str(x))
print('y_euler = ' + str(y_euler))
print('y_midpoint = ' + str(y_mittelpunkt))
print('y_mod_euler = ' + str(y_mod_euler))
print('y_runge_kutta  = ' + str(y_runge_kutta))

plt.figure(0)
plt.title('Numerische Lösung der verschiedenen Verfahren')
plt.plot(x, y_euler, label='Euler')
plt.plot(x, y_mittelpunkt, label='Mittelpunkt')
plt.plot(x, y_mod_euler, label='Mod. Euler')
plt.plot(x, y_runge_kutta, label='Runge-Kutta')
plt.legend()
plt.show()

plt.figure(1)
plt.title('Global Error')
plt.semilogy()
plt.plot(x, np.abs(y_euler - y_exact), label='Euler')
plt.plot(x, np.abs(y_mittelpunkt - y_exact), label='Mittelpunkt')
plt.plot(x, np.abs(y_mod_euler - y_exact), label='Mod. Euler')
plt.plot(x, np.abs(y_runge_kutta - y_exact), label='Runge-Kutta')
plt.legend()
plt.show()
