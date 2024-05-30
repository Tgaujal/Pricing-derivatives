import numpy as np
from scipy.stats import norm
def Eur_Binomial_model(S, K, r, T, sigma, option_type, N):
    # Calculate the time increment
    dt=T/N

    # Calculate the up and down factors
    u=np.exp(sigma*np.sqrt(dt))
    d=1/u

    # Calculata the risk-neutral probability
    p=(np.exp(r*dt)-d)/(u-d)

    # Initialize the stock price tree
    S_tree=np.zeros((N+1,N+1))
    S_tree[0,0]=S

    # Populate the stock price tree
    for j in range(N+1):
            S_tree[j,N]=S_tree[0,0]*u**j*d**(N-j)

    # Initialize the option price tree
    option_tree=np.zeros((N+1,N+1))

    # Populate the option price tree at maturity
    for j in range(N+1):
        if option_type=='call':
            option_tree[j,N]=max(0,S_tree[j,N]-K)
        else:
            option_tree[j,N]=max(0,K-S_tree[j,N])

    # Backward induction to calculate the option price at the initial node
    for i in range(N-1,-1,-1):
        for j in range(i+1):
            if option_type=='call':
                option_tree[j,i]=np.exp(-r*dt)*(p*option_tree[j+1,i+1]+(1-p)*option_tree[j,i+1])
            else:
                option_tree[j,i]=np.exp(-r*dt)*(p*option_tree[j+1,i+1]+(1-p)*option_tree[j,i+1])

    #Return the option price at the initial node
    return option_tree[0,0]


# AAPL data parameters with reduced steps
S = 150
K = 150
r = 0.05
sigma = 0.3
T = 0.5
N = 1000
option_type = 'call'

# Calculate the option price using the binomial model
binomial_price = Eur_Binomial_model(S, K, r, T, sigma, option_type, N)
print("Binomial Model Option Price (AAPL):", binomial_price)

# Calculate the option price using the Black-Scholes formula
d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
black_scholes_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
print("Black-Scholes Model Option Price (AAPL):", black_scholes_price)




