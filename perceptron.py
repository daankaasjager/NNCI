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


def train(data):
    sweep_count = 0
    while sweep_count != MAX_SWEEPS:
        for i in range(P):
            print(data[i])
            print("hi")
        print(sweep_count := sweep_count + 1)


if __name__ == "__main__":
    print("hello world")
    xi_vector_set = generate_dataset()
    train(xi_vector_set)