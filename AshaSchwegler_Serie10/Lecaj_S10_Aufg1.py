import numpy as np

def tf(f, a, b, n):
    sum = 0
    h = (b-a)/n
    for i in range(1, n):
        sum += f(a + i*h)
    return h * ((f(a) + f(b))/2 + sum)

def Lecaj_S10_Aufg1(f, a, b, m):
    n = m + 1
    T = np.zeros((n, n))
    for j in range(n):
        T[j,0] = tf(f, a, b, 2**j)
    for k in range(1,n):
        for j in range(m-k,-1,-1):
            T[j,k] = (4**k*T[j+1, k-1] - T[j,k-1]) / (4**k - 1)
    return T[0,m]

def f(v):
    return 97000 / ((-5) * v**2 - 570000)

# Aufgabe a)
print(Lecaj_S10_Aufg1(f, 0, np.pi, 4))

# Aufgabe b)
def f(v):
    return 97000 * v / ((-5) * v**2 - 570000)

print(Lecaj_S10_Aufg1(f, 100, 0, 4))