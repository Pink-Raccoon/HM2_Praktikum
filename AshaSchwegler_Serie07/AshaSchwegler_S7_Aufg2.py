import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

sp.init_printing()

# Aufgabe a)
x=np.array([2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. , 6.5, 7. , 7.5, 8. ,8.5, 9. , 9.5])

y=np.array([159.57209984, 159.8851819 , 159.89378952, 160.30305273,
       160.84630757, 160.94703969, 161.56961845, 162.31468058,
       162.32140561, 162.88880047, 163.53234609, 163.85817086,
       163.55339958, 163.86393263, 163.90535931, 163.44385491])


def f(x, p):
    return (p[0] + p[1] * (10 ** (p[2] + p[3] * x))) / (1 + 10 ** (p[2] + p[3] * x))

def gauss_newton_d(g, Dg, lam0, tol, max_iter, pmax, damping):
    k = 0
    lam = np.copy(lam0)
    increment = tol + 1
    err_func = np.linalg.norm(g(lam)) ** 2

    while increment > tol and k <= max_iter:
        # QR-Zerlegung von Dg(lam)
        [Q, R] = np.linalg.qr(Dg(lam))
        delta = np.linalg.solve(R, -Q.T @ g(lam)).flatten()  # Achtung: flatten() braucht es, um aus dem Spaltenvektor delta wieder
        # eine "flachen" Vektor zu machen, da g hier nicht mit Spaltenvektoren als Input umgehen kann
        # hier kommt die Däfmpfung, falls damping = 1
        p = 0
        while damping == 1 and p <= pmax and np.linalg.norm(g(lam + delta / (2 ** p))) ** 2 >= np.linalg.norm(g(lam)) ** 2:
            p += 1

        if p > pmax:
            p = 0

        # Update des Vektors Lambda
        lam = lam + delta / (2 ** p)
        err_func = np.linalg.norm(g(lam)) ** 2
        #xopt = scipy.optimize.fmin(err_func, lam0)
        increment = np.linalg.norm(delta)
        k = k + 1
        print('Iteration: ', k)
        print('lambda = ', lam)
        print('Inkrement = ', increment)
        print('Fehlerfunktional =', err_func)
    return lam, k

tol = 1e-5
max_iter = 30
pmax = 5
damping = 1

lam0 = np.array([100, 120, 3, -1], dtype=np.float64)
p = sp.symbols('p:{n:d}'.format(n=lam0.size))

g = sp.Matrix([y[k]-f(x[k], p) for k in range(len(x))])
Dg = g.jacobian(p)

g = sp.lambdify([p], g, 'numpy')
Dg = sp.lambdify([p], Dg, 'numpy')

[lam, n] = gauss_newton_d(g, Dg, lam0, tol, max_iter, pmax, damping)



t = sp.symbols('t')
F = f(t, lam)
F = sp.lambdify([t],F,'numpy')
t = np.linspace(x.min(), x.max())

plt.figure(1)
plt.plot(x, y, 'o')
plt.plot(t, F(t))
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Aufgabe b)
def gauss_newton(g, Dg, lam0, tol, max_iter):
    k=0
    lam=np.copy(lam0)
    increment = tol+1
    err_func = np.linalg.norm(g(lam))**2
    
    while tol <= err_func and max_iter<k : #Hier kommt Ihre Abbruchbedingung, die tol und max_iter berücksichtigen muss# 

        # QR-Zerlegung von Dg(lam) und delta als Lösung des lin. Gleichungssystems
        [Q,R] = np.linalg.qr(Dg(lam).flatten())
        [delta] = np.linalg.solve(R,-Q.T @ g(lam)).flatten()   
        
       
        # Update des Vektors Lambda        
        lam += delta.flatten()
        err_func = np.linalg.norm(g(lam)).flatten()**2
        increment = np.linalg.norm(delta)
        print('Iteration: ',k)
        print('lambda = ',lam)
        print('Inkrement = ',increment)
        print('Fehlerfunktional =', err_func)
    return(lam,k)

[lam_without,k] = gauss_newton(g,Dg,lam0,tol,max_iter)
t = sp.symbols('t')
F = f(t, lam_without)
F = sp.lambdify([t],F,'numpy')
t = np.linspace(x.min(), x.max())

plt.figure(2)
plt.plot(x, y, 'o')
plt.plot(t, F(t))
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# Es konvergiert .


# Aufgabe c)
def err_func(lam):
       return np.linalg.norm(g(lam)) ** 2


def err_func(lam):
       return np.linalg.norm(g(lam)) ** 2




