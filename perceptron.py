import numpy as np

# Global variables

# Number of dimensions of the input vectors
N = 20

# Scaler for the number of input vectors
ALPHA = 0.75

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
    return np.zeros((P, N))


def update_weight(weight_vector, input_vector, vector_label):
    weight_vector += (1/N)*input_vector*vector_label


def train(data):
    sweep_count = 0
    w = initialize_weights()
    while sweep_count != MAX_SWEEPS:
        no_updates = True
        for i in range(P):
            if np.dot(w[i], data[i][0]) * data[i][1] <= 0:
                update_weight(w[i], data[i][0], data[i][1])
                no_updates = False
        # Unchanged for a whole sweep of the weight vectors
        if no_updates:
            print("Finished training")
            break
        print("sweepcount: ")
        print(sweep_count := sweep_count + 1)


if __name__ == "__main__":
    print("hello world")
    xi_vector_set = generate_dataset()

    train(xi_vector_set)