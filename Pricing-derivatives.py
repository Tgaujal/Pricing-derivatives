import numpy as np
from scipy.stats import norm
def Binomial_model(S, K, r, T, sigma, option_type, N):
    dt=T/N
    u=np.exp(sigma*np.sqrt(dt))
    d=1/u
    p=(np.exp(r*dt)-d)/(u-d)
    S_tree=np.zeros((N+1,N+1))
    S_tree[0,0]=S
    for i in range(1,N+1):
        for j in range(i+1):
            S_tree[j,i]=S_tree[0,0]*u**j*d**(i-j)
    option_tree=np.zeros((N+1,N+1))
    for j in range(N+1):
        if option_type=='call':
            option_tree[j,N]=max(0,S_tree[j,N]-K)
        else:
            option_tree[j,N]=max(0,K-S_tree[j,N])
    for i in range(N-1,-1,-1):
        for j in range(i+1):
            if option_type=='call':
                option_tree[j,i]=max(0,np.exp(-r*dt)*(p*option_tree[j,i+1]+(1-p)*option_tree[j+1,i+1]))
            else:
                option_tree[j,i]=max(0,np.exp(-r*dt)*(p*option_tree[j,i+1]+(1-p)*option_tree[j+1,i+1]))
    return option_tree[0,0]
S=100
K=100
r=0.06
sigma=0.3
T=1
N=10000
option_type='call'
print(Binomial_model(S, K, r, T, sigma, option_type, N))
C=S*norm.cdf((np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T)))-K*np.exp(-r*T)*norm.cdf((np.log(S/K)+(r-0.5*sigma**2)*T)/(sigma*np.sqrt(T)))
print(C)
