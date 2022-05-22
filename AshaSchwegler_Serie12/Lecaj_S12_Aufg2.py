import numpy as np
import matplotlib.pyplot as plt


# Aufgabe 2c)
def f(x, y):
    return 1 - (y/x)

def Lecaj_S12_Aufg1(f, x, h, y0):
    s = 4
    b = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    c = np.array([0, 0.25, 0.5, 0.75], dtype=np.float64)
    a = np.array([[0, 0, 0, 0],[0.5, 0, 0, 0],[0, 0.75, 0, 0],[0, 0, 1, 0]], dtype=np.float64)

    y = np.full(x.shape[0], 0, dtype=np.float64)
    y[0] = y0

    for i in range(x.size - 1):
        k = np.full(s, 0, dtype=np.float64)

        for n in range(s):
            k[n] = f(x[i] + (c[n] * h), y[i] + h * np.sum([a[n][m] * k[m] for m in range(n - 1)]))

        y[i + 1] = y[i] + h * np.sum([b[n] * k[n] for n in range(s)])

    return y

def exactSolution(x):
    return (x/2) + (9/(2*x))

a = 1
b = 6
n = 500
y0 = 5
h = h = ((b-a) * 1.0) / n

x = np.arange(a, b + h, h, dtype=np.float64)

y = Lecaj_S12_Aufg1(f,x,h,y0)

h = ((b-a) * 1.0) / n
x_exact = np.arange(a, b+h, h, dtype=np.float64)
y_exact = exactSolution(x)

plt.figure(0)
plt.title("Rungakutta, exakt")
plt.plot(x, y, label='Runge-Kutta')
plt.plot(x_exact, y_exact, label='Exakt')
plt.legend()
plt.show()

plt.figure(1)
plt.title('Global error')
plt.semilogy()
plt.plot(x, np.abs(y - y_exact, label='Runge-Kutta-'))
plt.legend()
plt.show()