import numpy as np
from scipy.stats import norm

def AME_Binomial_model(S0, K, T, r, sigma, option_type, N):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Build the price tree for the underlying asset
    S = np.zeros((N + 1, N + 1))
    S[0, 0] = S0
    for i in range(1, N + 1):
        for j in range(i + 1):
            S[j, i] = S[0, 0] * u ** j * d ** (i - j)

    # Initialize option values at maturity
    V = np.zeros((N + 1, N + 1))
    for j in range(N + 1):
        if option_type == 'call'
            V[j, N] = max(0, S[j, N] - K)
        else:
            V[j,N] = max(0, K - S[j,N])
    
    # Backward induction to evaluate the option
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            if option_type == 'call'
                continuation_value = (p * V[j + 1, i + 1] + (1 - p) * V[j, i + 1]) * np.exp(-r * dt)
                exercise_value = max(S[j, i] - K, 0)
                V[j, i] = max(continuation_value, exercise_value)
            else :
                continuation_value = (p * V[j + 1, i + 1] + (1 - p) * V[j, i + 1]) * np.exp(-r * dt)
                exercise_value = max(K - S[j, i], 0)
                V[j, i] = max(continuation_value, exercise_value)
    return V[0, 0]

# Parameters for AAPL stock
S0_aapl = 150
T_aapl = 1
r_aapl = 0.04
sigma_aapl = 0.3
N_aapl = 1000
option_type = 'call'

# Deep In-The-Money option parameters
K_deep_in_the_money = 100

# Calculation with the binomial model for deep in-the-money option
binomial_value_deep_in = AME_Binomial_model(S0_aapl, K_deep_in_the_money, T_aapl, r_aapl, sigma_aapl,option_type, N_aapl)

# Comparison with the Black-Scholes model for deep in-the-money option
d1_deep_in = (np.log(S0_aapl / K_deep_in_the_money) + (r_aapl + 0.5 * sigma_aapl ** 2) * T_aapl) / (sigma_aapl * np.sqrt(T_aapl))
d2_deep_in = d1_deep_in - sigma_aapl * np.sqrt(T_aapl)
black_scholes_value_deep_in = S0_aapl * norm.cdf(d1_deep_in) - K_deep_in_the_money * np.exp(-r_aapl * T_aapl) * norm.cdf(d2_deep_in)

# Deep Out-of-The-Money option parameters
K_deep_out_of_the_money = 200

# Calculation with the binomial model for deep out-of-the-money option
binomial_value_deep_out = AME_Binomial_model(S0_aapl, K_deep_out_of_the_money, T_aapl, r_aapl, sigma_aapl, N_aapl)

# Comparison with the Black-Scholes model for deep out-of-the-money option
d1_deep_out = (np.log(S0_aapl / K_deep_out_of_the_money) + (r_aapl + 0.5 * sigma_aapl ** 2) * T_aapl) / (sigma_aapl * np.sqrt(T_aapl))
d2_deep_out = d1_deep_out - sigma_aapl * np.sqrt(T_aapl)
black_scholes_value_deep_out = S0_aapl * norm.cdf(d1_deep_out) - K_deep_out_of_the_money * np.exp(-r_aapl * T_aapl) * norm.cdf(d2_deep_out)

# Print results
print("Deep In-The-Money Option:")
print("Binomial Model (American):", binomial_value_deep_in)
print("Black-Scholes Model (European):", black_scholes_value_deep_in)

print("\nDeep Out-of-The-Money Option:")
print("Binomial Model (American):", binomial_value_deep_out)
print("Black-Scholes Model (European):", black_scholes_value_deep_out)
