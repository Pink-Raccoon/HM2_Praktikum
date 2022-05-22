import numpy as np
import matplotlib.pyplot as plt

# Aufgabe 1)
def f(x, y):
    return 1 - (y/x)

def Lecaj_S12_Aufg1(f, a, b, n, y0):
    h = ((b-a) * 1.0) / n
    x = np.arange(a, b+h, h, dtype=np.float64)
    y = np.full(x.shape[0], 0, dtype=np.float64)
    y[0] = y0

    for i in range(x.size - 1):
        k1 = f(x[i], y[i])
        #print("K1:", k1)
        k2 = f(x[i] + (h / 2.0), y[i] + (h / 2.0) * k1)
        #print("K2:", k2)
        k3 = f(x[i] + (h / 2.0), y[i] + (h / 2.0) * k2)
        #print("K3:", k3)
        k4 = f(x[i] + h, y[i] + h * k3)
        #print("K4:", k4)
        y[i + 1] = y[i] + h * (1 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        #print(y[i+1])

    return x,y

def exactSolution(x):
    return (x/2) + (9/(2*x))

a = 1
b = 6
n = 500
y0 = 5

x, y = Lecaj_S12_Aufg1(f,a,b,n,y0)


#Aufgabe 2a)
h = ((b-a) * 1.0) / n
x_exact = np.arange(a, b+h, h, dtype=np.float64)
y_exact = exactSolution(x)

plt.figure(0)
plt.title("Klassische Runga-Kutta Verfahren vs. Exakte Lösung")
plt.plot(x, y, label='Runge-Kutta (Numerisch)')
plt.plot(x_exact, y_exact, label='Exakt')
plt.legend()
plt.show()


plt.figure(1)
plt.title("Exakte Lösung")
plt.plot(x_exact, y_exact, label='Exakt')
plt.legend()
plt.show()
