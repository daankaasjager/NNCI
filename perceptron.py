import numpy as np

# Global variables

# Number of dimensions of the input vectors
N = 40

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
    return np.zeros(N)


def update_weight(weight_vector, input_vector, vector_label, E):
    if E <= 0:
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
            # print(w)
            # print(sweep_count)
            return True
        sweep_count += 1
    return False    


if __name__ == "__main__":
    print("hello world")
    while ALPHA != 3:
        P = int(ALPHA * N)
        num_succesful = 0
        for i in range(ND):
            xi_vector_set = generate_dataset()
            successful = train(xi_vector_set)
            if successful:
                num_succesful += 1
        ALPHA += 0.25
        print("num_succesful: ", num_succesful)
        num_succesful /= ND
        print("alpha: ", ALPHA)
