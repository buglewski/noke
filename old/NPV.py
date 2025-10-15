import numpy as np
import random 


alpha = 0.5
EP = 1
r = 0.1
P0 = 3

#Z = P0
for it in range(0, 1):
    Z = P0
    for i in range(1, 1000):
        Z += ((alpha**i) * P0 + (1-alpha**i) * EP)/((1+r)**i)
    print(f"iteration{it} : {Z}0")

Z = (1+r)*P0/(1+r-alpha) + EP/(r) - alpha*EP/(1+r-alpha)
print(Z)

Res = []
for it in range(0, 100):
    Z = P0
    P = P0
    for i in range(1, 100):
        al = random.randint(1,100)
        if al > 100*alpha:
            P = random.randint(1,200)/100
        Z += P/(1+r)**i
    Res += [Z]
    #print(f"iteration{it} : {Z}0")

print(np.mean(Res))

m = 1000
sigma = 1

for it in range(10):
    c = np.random.normal(loc = 1, scale = 1, size = m)
    i = random.randint(0, m-1)
    EP = 1/8*((2+sigma*np.sum(c))/(2+m*sigma)-c[i])**2
    print(f"iteration{it}: {EP}")

EP = 1/8*((2+sigma*m)/(2+sigma*m)-1)**2
print(EP)