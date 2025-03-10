import numpy as np
import scipy.stats as stats

# Defining required parameters 
range_a = (0.5, 2)  # Boundary separation (a)
range_v = (0.5, 2)  # Drift rate (v)
range_t = (0.1, 0.5)  # Nondecision time (t)

# Defining necessary equations:

# Forward EZ equations  
def forward_equations(v, a, t):
    y = np.exp(-v * a)
    R_pred = 1 / (y + 1)
    M_pred = t + (a / (2 * v)) * (1 - y) / (1 + y)
    V_pred= (a / (2 * v**3)) * ((1 - 2 * v * a * y - y**2) / (y + 1)**2)
    return R_pred, M_pred, V_pred

# Observed summary statictics equations
def simulate_observed(R_pred, M_pred, V_pred, N):
    
    epsilon = 1e-6
    R_pred = np.clip(R_pred, epsilon, 1 - epsilon)

    T_obs = np.random.binomial(N, R_pred)
    M_obs = np.random.normal(M_pred, np.sqrt(max(V_pred, epsilon) / N))
    if N > 1:
        V_obs = np.random.gamma((N - 1) / 2, 2 * max(V_pred, epsilon) / (N - 1))
    else:
        V_obs = V_pred  

    return T_obs / N, M_obs, V_obs

# Inverse EZ Equations 
def inverse_equations(T_obs, M_obs, V_obs):

    epsilon = 1e-6  
    L = np.log((T_obs + epsilon) / (1 - T_obs + epsilon))
   
    #ChatGPT Acknowledgement: Epsilon codes pulled form ChatGPT to help with extreme values due to error messages 

    v_est = np.sign(T_obs - 0.5) * 4 * np.sqrt(L * (T_obs**2 * L - T_obs * L + T_obs - 0.5) / V_obs)
    a_est = L / v_est
    t_est = M_obs - (a_est / (2 * v_est)) * (1 - np.exp(-v_est * a_est)) / (1 + np.exp(-v_est * a_est))
    return v_est, a_est, t_est


# Run Simulation
def simulate_and_recover(N, iterations=1000):
    biases = []
    squared_errors = []
    
    for _ in range(iterations):
        # Selecting some true parameters 
        v_true = np.random.uniform(*range_v)
        a_true = np.random.uniform(*range_a)
        t_true = np.random.uniform(*range_t)
        
        # Generating predicted statistics
        R_pred, M_pred, V_pred = forward_equations(v_true, a_true, t_true)
        
        # Simulate observed statistics
        T_obs, M_obs, V_obs = simulate_observed(R_pred, M_pred, V_pred, N)
        
        # Estimated parameters
        v_est, a_est, t_est = inverse_equations(T_obs, M_obs, V_obs)
        
        # Compute bias and squared error
        bias = np.array([v_true - v_est, a_true - a_est, t_true - t_est])
        squared_error = bias ** 2
        biases.append(bias)
        squared_errors.append(squared_error)
    
    # Compute averages
    mean_bias = np.mean(biases, axis=0)
    mean_squared_error = np.mean(squared_errors, axis=0)
    
    return mean_bias, mean_squared_error

# Run for N = 10, 40, 4000 
if __name__ == "__main__":
    with open("README.md", "a") as f:  
        f.write("\n## Simulation Results\n")
        for N in [10, 40, 4000]:
            bias, mse = simulate_and_recover(N)
            results_str = f"N={N} | Bias: {bias} | MSE: {mse}\n"
            print(results_str)  # Print to terminal
            f.write(results_str)  # Append to README.md

#ChatGPT Acknowledgment:  
# Used software to print results directly into README file 