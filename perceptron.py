import numpy as np

# Global variables
P = 10
N = 20
ALPHA = 0.75
MAX_SWEEPS = 3


def generate_dataset():
    mean = 0
    variance = 1
    xi_vector_set = [[np.random.normal(mean, np.sqrt(variance), N), np.random.choice([-1, 1])] for _ in range(P)]
    return xi_vector_set


def initialize_weights():
    return np.zeros((P, N))


def calculate_Ev():
    pass


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

        if no_updates:
            print("Finished training")
            break
        print("sweepcount: ")
        print(sweep_count := sweep_count + 1)


if __name__ == "__main__":
    print("hello world")
    xi_vector_set = generate_dataset()
    train(xi_vector_set)