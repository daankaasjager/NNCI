import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

# Global variables

# Number of dimensions of the input vectors
N = 20

# Scaler for the number of input vectors
ALPHA = 0.75
ALPHA_INCREMENT = 0.25
ALPHA_MAX_RANGE = 3
# Threshold value
C = 0.1

# Number of input vectors
P = int(ALPHA * N)

# Iterations of the algorithms
ND = 50

# Nmax in the assignment(
MAX_SWEEPS = 100


def generate_dataset():
    mean = 0
    variance = 1
    xi_vector_set = [[np.random.normal(mean, np.sqrt(variance), N), np.random.choice([-1, 1])] for _ in range(P)]
    return xi_vector_set


def initialize_weights():
    return np.zeros(N)


# Function to calculate the theoretical probability of linear separability
def Pls(P, N):
    if P <= N:
        return 1
    else:
        sum_combinations = sum(comb(P - 1, i, exact=True) for i in range(N))
        return (2 ** (1 - P)) * sum_combinations


def update_weight(weight_vector, input_vector, vector_label, E):
    if E <= C:
        weight_vector += (1/N)*input_vector*vector_label
    return weight_vector


def check_convergence(w, data):
    for i in range(P):
        E = np.dot(w, data[i][0]) * data[i][1]
        if E <= 0:
            return False
    return True


def train(data):
    sweep_count = 0
    w = initialize_weights()
    while sweep_count != MAX_SWEEPS:
        for input_vector, vector_label in data:
            E = np.dot(w, input_vector) * vector_label
            w = update_weight(w, input_vector, vector_label, E)
        if check_convergence(w, data):
            return True
        sweep_count += 1
    return False    


def plot_graph(alpha_values, Qls_values, Pls_values):
    plt.figure(figsize=(10, 6))

    # Plot experimental success rate
    plt.plot(alpha_values, Qls_values, 'o-', label='Experimental Ql.s.')

    # Plot theoretical probability
    plt.plot(alpha_values, Pls_values, 's-', label='Theoretical Pl.s.')

    # Annotating the graph
    plt.title(f'Comparison of Experimental Success Rate and Theoretical Probability (N = {N}, nd = {ND},'
              f' nmax = {MAX_SWEEPS}, c = {C})')
    plt.xlabel('Alpha (P/N)')
    plt.ylabel('Success Rate / Theoretical Probability')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show() 


if __name__ == "__main__":
    print("hello world")
    alpha_values = []
    Qls_values = []
    Pls_values = []
    while ALPHA <= ALPHA_MAX_RANGE:
        P = int(ALPHA * N)
        num_successful = 0
        for i in range(ND):
            xi_vector_set = generate_dataset()
            successful = train(xi_vector_set)
            if successful:
                num_successful += 1
        success_rate = num_successful / ND
        Qls_values.append(success_rate)
        alpha_values.append(ALPHA)
        
        theoretical_probability = Pls(P, N)
        Pls_values.append(theoretical_probability)
        
        print(f"alpha: {ALPHA}, num_successful: {num_successful}, success_rate: {success_rate}, Pl.s.: {theoretical_probability}")
        
        ALPHA += ALPHA_INCREMENT
    
    plot_graph(alpha_values, Qls_values, Pls_values)