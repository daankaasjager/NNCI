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
ND = 10

# Nmax in the assignment(
MAX_SWEEPS = 100

# Stopping criterion for the angular change
STOPPING_ANGLE = 0.02
# Determines whether to run the angular stopping or a normal run
ANGULAR_STOP = True


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
    return weight_vector + (1 / N) * xi_vector * vector_label


def calculate_generalization_error(w, w_star):
    # Formula from the assignment brief
    return 1 / np.pi * np.arccos(np.dot(w, w_star) / (np.linalg.norm(w) * np.linalg.norm(w_star)))


def train(data):
    sweep_count = 0
    w = initialize_weights()
    angular_change = None

    while sweep_count != MAX_SWEEPS * P:
        # Calculating the stabilities
        stabilities = [label * np.dot(w, xi_vector) for xi_vector, label in data]

        # Avoiding division by zero
        if sweep_count != 0:
            stabilities /= np.linalg.norm(w)

        # Finding the minimum stability
        min_stability_index = np.argmin(stabilities)
        xi_min, label_min = data[min_stability_index]

        old_weight_copy = w

        # Updating the weight vector
        w = update_weight(w, xi_min, label_min)

        if ANGULAR_STOP:
            # Angular change stopping criterion
            if not np.all(w == 0) and not np.all(old_weight_copy == 0):
                angular_change = np.arccos(
                    np.dot(w, old_weight_copy) / (np.linalg.norm(w) * np.linalg.norm(old_weight_copy))) / np.pi

            if (angular_change is not None) and (angular_change < STOPPING_ANGLE):
                print(
                    f"Angular change {angular_change} smaller than threshold: {STOPPING_ANGLE} after {sweep_count} sweeps. Training terminated")
                break
        sweep_count += 1
    return w, sweep_count


if __name__ == "__main__":
    print("hello world")
    alpha_values = []
    generalization_errors = []
    avg_generalization_errors = []
    run_count = 0
    avg_angular_sweep_count = []
    while ALPHA <= 5.0:
        all_angular_sweep_count = []
        P = int(ALPHA * N)
        for _ in range(ND):
            # Generating the teacher perceptron
            w_star = generate_teacher_perceptron()

            # Generating the dataset and training with minover, returning the final weight vector
            xi_vector_set = generate_dataset(w_star)
            final_w, sweep_count = train(xi_vector_set)

            # Getting the generalization error
            generalization_error = calculate_generalization_error(final_w, w_star)
            generalization_errors.append(generalization_error)

            # Get the sweep_count for each run
            all_angular_sweep_count.append(sweep_count)

        # Calculating the average generalization error for graphing
        alpha_values.append(ALPHA)
        avg_error = np.mean(generalization_errors)
        avg_generalization_errors.append(avg_error)
        avg_angular_sweep_count.append(np.mean(all_angular_sweep_count))
        if run_count == 0:
            ALPHA += 0.25
        else:
            ALPHA += 0.5
        run_count += 1

    directory = 'D:/Master/blok_1b/NNCI/assignment2/images/'
    learning = f'learning_{STOPPING_ANGLE:.2f}.png'
    sweeps = f'sweeps_{STOPPING_ANGLE:.2}.png'
    # Plotting the final graph
    plt.plot(alpha_values, avg_generalization_errors, marker='o')
    plt.xlabel('\u03B1 (P/N)')
    plt.ylabel('Average Generalization Error \u03B5g(tmax)')
    plt.savefig(directory + learning)
    plt.show()

    plt.plot(alpha_values, avg_angular_sweep_count, marker='o')
    plt.xlabel('\u03B1 (P/N)')
    plt.ylabel(f'Average number of sweeps for \u03b8 = {STOPPING_ANGLE:.2e}')
    plt.savefig(directory + sweeps)
    plt.show()
