import numpy as np
import pandas as pd
import random

class ApsundNokeModelFC:
    def __init__(self, distr, data, n_depth = 20000):
        self.distr = distr
        self.data = data
        self.n_depth = n_depth
        self.n_epochs = data.shape[0]
        self.CF = 0.01

    def EP_ent(self, FC, epoch):
        c = self.data["c"][epoch]
        S = self.data["S"][epoch]
        sigma = self.data["sigma"][epoch]
        m = len(FC) * self.CF
        EFC = (self.data["a"][epoch] + self.data["b"][epoch])/2
        return (S/8 * ((1-c)/(1+sigma*m))**2 - EFC)

    def NPV_for_entrant(self, FC, epoch):
        r = self.data["r"][epoch]
        c = self.data["c"][epoch]
        S = self.data["S"][epoch]
        sigma = self.data["sigma"][epoch]
        n = len(FC) * self.CF
        EFC = (self.data["a"][epoch] + self.data["b"][epoch])/2
        return (1+r)/r*(S/8*((1-c)/(1+sigma*n))**2-EFC)
        
    def NPV_exit(self, FC, FC_i, epoch):
        r = self.data["r"][epoch]
        c = self.data["c"][epoch]
        S = self.data["S"][epoch]
        sigma = self.data["sigma"][epoch]
        n = len(FC) * self.CF
        alpha = self.data["alpha"][epoch]
        EFC = (self.data["a"][epoch] + self.data["b"][epoch])/2
    
        return ((1+r)/r* S/8*(((1-c)/(1+sigma*n))**2)
                -(1+r)/(1+r-alpha)*FC_i
                -(1/r-alpha/(1+r-alpha))*EFC)
    
    def scenario(self, out = False):
        #data prepare
        NPV_ent = [0] * self.n_epochs
        M = [0] * self.n_epochs
        P_mean = [0] * self.n_epochs
        EP_ = [0] * self.n_epochs
        C_mean = [0] * self.n_epochs
        C_max = [0] * self.n_epochs
        C_min_ex = [0] * self.n_epochs
        Sigma2 = [0] * self.n_epochs
        M_ent = [0] * self.n_epochs
        M_ex = [0] * self.n_epochs
        _EP_ent = [0] * self.n_epochs

        X = [0] * self.n_epochs
        P = [0] * self.n_epochs

        #zero epoch
        FC = np.random.uniform(self.data['a'][0], self.data['b'][0], size = 1)
        
        for epoch in range(self.n_epochs):
            #loading data
            a = self.data['a'][epoch]
            b = self.data['b'][epoch]
            c = self.data["c"][epoch]
            r = self.data["r"][epoch]
            ef = self.data["ef"][epoch]
            alpha = self.data["alpha"][epoch]
            S = self.data["S"][epoch]

            NPV1 = self.NPV_for_entrant(FC, epoch)
            #print(self.NPV_for_entrant(FC, epoch))
            #entrance stage
            n_ent = 0
            while NPV1 > ef:
                f = np.random.uniform(a, b)
                FC = np.append(FC, f)
                n_ent += 1
                NPV1 = self.NPV_for_entrant(FC, epoch)

            #recounting stage
            for i in range(len(FC)):
                ar = random.random()
                if ar > alpha:
                    FC[i] = np.abs(np.random.uniform(a, b))
            
            # sort of C
            FC = np.flip(np.sort(FC))

            # exit stage
            n_ex = 0
            i = 0
            C_min_exited = FC[0]
            while i < len(FC):
                if self.NPV_exit(FC, FC[i], epoch)<0:
                    C_min_exited = FC[i]
                    FC = np.delete(FC, i)
                    n_ex+=1
                else:   
                    i += 1

            NPV_ent[epoch] = NPV1
            M[epoch] = len(FC) * self.CF
            #P_mean[epoch] = np.mean(P_fact)
            #EP_[epoch] = EP
            C_mean[epoch] = np.mean(FC)
            C_max[epoch] = np.amax(FC)
            C_min_ex[epoch] = C_min_exited
            #Sigma2[epoch] = sigma2
            M_ent[epoch] = n_ent * self.CF
            M_ex[epoch] = n_ex * self.CF

            X[epoch] = 1/4 * (1-c)/(1+self.data["sigma"][epoch]*M[epoch])
            P[epoch] = 1-2*X[epoch]-2*self.data["sigma"][epoch] * M[epoch] * X[epoch]

            if out:
                print(f"epoch{epoch}, number of firms: {M[epoch]}, entrats: {M_ent[epoch]}, exiters: {M_ex[epoch]}, C_mean: {C_mean[epoch]}")
        
        self.data["NPV_ent"] = NPV_ent
        self.data["M"] = M 
        #self.data["Pr_mean"] = P_mean
        #self.data["EP"] = EP_
        self.data["C_mean"] = C_mean
        self.data["C_max"] = C_max
        self.data["C_min_ex"] = C_min_ex
        #self.data["Sigma2"] = Sigma2
        self.data["K"] = M_ent
        #self.data["_EP_ent"] = _EP_ent
        self.data["X"] = X
        self.data["P"] = P

    def getData(self):
        return self.data
    
if __name__ == '__main__':
    a = 0.1
    b = 0.2

    def _P(f, a=a, b=b):
        return (f-a)/(b-a)
    
    def _p(f, a=a, b=b):
        return 1/(b-a)
    
    Ef = (a+b)/2
    #parameters of utility
    sigma = 1
    S = 16

    #parameters of firms
    

    c = 0.1 # costs
    ef = 0.05 # entrance costs
    r = 0.1 #return of interest
    alpha = 0.9

    n = 100

    data = {"a": [a]*n,
            "b": [b]*n,
            "sigma": [sigma]*n,
            "S": [S]*n,
            "c": [c]*n,
            "ef": [ef]*n,
            "r": [r]*n,
            "alpha": [alpha]*n}

    df = pd.DataFrame(data)

    Noke = ApsundNokeModelFC(distr = np.random.uniform,
                       data = df)
    
    Noke.scenario(out=False)

    result = Noke.getData()

    Cmax = result["C_max"].mean()
    Cminex = result["C_min_ex"].mean()
    K = result["K"].mean()
    M = result["M"].mean()
    F = Cmax * 3/3 + Cminex * 0/3
    print(result.describe())
    print(f"K={K}, M={M}, F={F}")

    print(
        (1+r)/r *( S/8 * ((1-c)/(1+sigma*(M+K)))**2 - Ef) - ef,
        (1+r)/r * S/8 *((1-c)/(1+sigma*M))**2 - (1/r - alpha/(1+r-alpha)) * Ef - (1+r)/(1+r-alpha) * F,
        (1-alpha)*(1-_P(F)) * M - K * _P(F)
    )

    P = _P(F)/(1-alpha+alpha*_P(F))
    A = (S/8 * (1+r - alpha) / ((1-alpha)*Ef + r*F)) ** 0.5
    B = (S/8 * (1+r)/(r*ef + (1+r)*Ef))**0.5
    
    print(P, A, B)
    print(
        "M + K\n",
        M + K,
        M/P,
        1/sigma * ((1-c)*B - 1)
    )
    print(
        "M\n",
        1/sigma * ((1-c) * A - 1),
        1/sigma * ((1-c)*B - 1)*P
    )
    #(1-c)(A-B*P) = 1 - P
    c_r = 1 - (1-P)/(A-P*B)
    print(c_r)