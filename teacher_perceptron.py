import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

# Global variables

# Number of dimensions of the input vectors
N = 40

# Scaler for the number of input vectors
ALPHA = 0.25

# Number of input vectors
P = int(ALPHA * N)

# Iterations of the algorithms
ND = 50

# Nmax in the assignment(
MAX_SWEEPS = 100


def generate_teacher_perceptron():
    w_star = np.random.randn(N)
    w_star /= np.linalg.norm(w_star)
    w_star *= np.sqrt(N)
    # if (int(np.linalg.norm(w_star)**2)) != N:
    #     print(int(np.linalg.norm(w_star)**2))
    #     print("Error: Teacher perceptron not generated correctly")
    #     exit()
    return w_star


def generate_dataset(w_star):
    dataset = []
    for _ in range(P):
        xi_vector = np.random.randn(N)
        # Sign calculation from the assignment brief
        sign = np.sign(np.dot(w_star, xi_vector))
        dataset.append([xi_vector, sign])
    return dataset


def initialize_weights():
    return np.zeros(N)


def update_weight(weight_vector, xi_vector, vector_label):
    # Formula from the assignment brief
    return weight_vector + (1/N)*xi_vector*vector_label


def calculate_generalization_error(w, w_star):
    # Formula from the assignment brief
    return 1/np.pi * np.arccos(np.dot(w, w_star)/(np.linalg.norm(w) * np.linalg.norm(w_star)))


def train(data):
    sweep_count = 0
    w = initialize_weights()
    while sweep_count != MAX_SWEEPS*P:
        # Calculating the stabilities
        stabilities = [label * np.dot(w, xi_vector) for xi_vector, label in data]
        
        # Avoiding division by zero
        if sweep_count != 0:
            stabilities /= np.linalg.norm(w)

        # Finding the minimum stability
        min_stability_index = np.argmin(stabilities)
        xi_min, label_min = data[min_stability_index]

        # Updating the weight vector
        w = update_weight(w, xi_min, label_min)

        sweep_count += 1
    return w        


if __name__ == "__main__":
    print("hello world")
    alpha_values = []
    generalization_errors = []
    avg_generalization_errors = []
    while ALPHA != 5:
        P = int(ALPHA * N)
        for _ in range(ND):
            # Generating the teacher perceptron
            w_star = generate_teacher_perceptron()
            
            # Generating the dataset and training with minover, returning the final weight vector
            xi_vector_set = generate_dataset(w_star)
            final_w = train(xi_vector_set)
            
            # Getting the generalization error
            generalization_error = calculate_generalization_error(final_w, w_star)
            generalization_errors.append(generalization_error)

        # Calculating the average generalization error for graphing
        alpha_values.append(ALPHA)
        avg_error = np.mean(generalization_errors)
        avg_generalization_errors.append(avg_error)    

        ALPHA += 0.25
    
    # Plotting the final graph    
    plt.plot(alpha_values, avg_generalization_errors, marker='o')
    plt.xlabel('Alpha (P/N)')
    plt.ylabel('Average Generalization Error')
    plt.title('Learning Curve')
    plt.show()        
    
