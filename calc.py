import numpy as np

import matplotlib.pyplot as plt

plt.style.use('_mpl-gallery')




a_ = 0.1
b_ = 0.2

def P_(f, a = a_, b = b_):
    return (f-a)/(b-a)

def p_(f, a=a_, b=b_):
    return 1/(b-a)

Ef = (a_ + b_)/2

S = 8

X, Y, Z = [], [], []

eps = 0.01
while eps < 0.8:
    alpha = 0.5
    while alpha < 0.91:
        r = 0.1
        while r < 0.4:
            f = 0.15
            #print(f"eps = {eps}. alpha={alpha}, r={r}")
            while f < 0.2:
                P = P_(f) / (1 - alpha + alpha * P_(f))
                A = (S/8 * (1+r - alpha) / ((1-alpha)*Ef + r*f)) ** 0.5
                B = (S/8 * (1+r)/(r*eps + (1+r)*Ef))**0.5
                Pd = (1-alpha)*p_(f) / (1 - alpha + alpha * P_(f)) ** 2
                Ad = - (r/2) * (1+r-alpha)**0.5 / ((1-alpha)*Ef + r*f) ** 1.5
                check = - (Pd * (B - A) + Ad * (P-1)) / (A-P*B)**2 #производная вся
                c = (1 - (1-P)/(A-P*B)) # сами косты
                M = ((1-c) * A - 1)
                K = M / P - M
                if check > 0 and M > 0 and K > 0:
                    print(c, '\t', check, '\t', f"F={f}, M={M}, K={K}")
                X.append(eps)
                Y.append(alpha)
                Z.append(check)
                f += 0.01
            
            r += 0.05
        alpha += 0.02
    eps += 0.03

# Plot
"""
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X, Y, Z)
ax.set_xlabel('epsilon')
ax.set_ylabel('alpha')
ax.set_zlabel('dc/df')

plt.show()

"""