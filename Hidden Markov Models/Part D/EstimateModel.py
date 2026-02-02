import sys
import math
import time

def read_matrix(input_line):
    """Reads a matrix from a string input."""
    data = list(map(float, input_line.split()))
    rows, cols = int(data[0]), int(data[1])
    matrix = []
    idx = 2
    for _ in range(rows):
        matrix.append(data[idx:idx + cols])
        idx += cols
    return matrix

def read_observations(input_line):
    """Reads the sequence of observations from a string input."""
    data = list(map(int, input_line.split()))
    return data[1:]

def forward_algorithm_scaled(A, B, pi, observations):
    """Calculates the forward probabilities with scaling to prevent underflow."""
    num_states = len(pi[0])
    T = len(observations)
    alpha = [[0] * num_states for _ in range(T)]
    scalers = [0] * T

    # Initialization
    for i in range(num_states):
        alpha[0][i] = pi[0][i] * B[i][observations[0]]
    scalers[0] = 1 / sum(alpha[0])
    alpha[0] = [alpha[0][i] * scalers[0] for i in range(num_states)]

    # Recursion
    for t in range(1, T):
        for j in range(num_states):
            alpha[t][j] = sum(alpha[t - 1][i] * A[i][j] for i in range(num_states)) * B[j][observations[t]]
        scalers[t] = 1 / sum(alpha[t])
        alpha[t] = [alpha[t][i] * scalers[t] for i in range(num_states)]

    return alpha, scalers

def backward_algorithm_scaled(A, B, observations, scalers):
    """Calculates the backward probabilities with scaling."""
    num_states = len(A)
    T = len(observations)
    beta = [[0] * num_states for _ in range(T)]

    # Initialization
    for i in range(num_states):
        beta[T - 1][i] = scalers[T - 1]

    # Recursion
    for t in range(T - 2, -1, -1):
        for i in range(num_states):
            beta[t][i] = sum(A[i][j] * B[j][observations[t + 1]] * beta[t + 1][j] for j in range(num_states))
        beta[t] = [beta[t][i] * scalers[t] for i in range(num_states)]

    return beta

def compute_log_likelihood(scalers):
    """Calculates the log-likelihood of the observation sequence."""
    return -sum(math.log(s) for s in scalers)

def baum_welch(A, B, pi, observations, max_time=0.8, tolerance=1e-4):
    """Estimates the model parameters using the Baum-Welch algorithm."""
    num_states = len(A)
    num_emissions = len(B[0])
    T = len(observations)
    start_time = time.time()

    previous_log_likelihood = float("-inf")
    while time.time() - start_time < max_time:
        # E-step: Compute alpha, beta, gamma, and xi
        alpha, scalers = forward_algorithm_scaled(A, B, pi, observations)
        beta = backward_algorithm_scaled(A, B, observations, scalers)
        log_likelihood = compute_log_likelihood(scalers)

        gamma = [[0] * num_states for _ in range(T)]
        xi = [[[0] * num_states for _ in range(num_states)] for _ in range(T - 1)]

        for t in range(T - 1):
            denom = sum(
                alpha[t][i] * A[i][j] * B[j][observations[t + 1]] * beta[t + 1][j]
                for i in range(num_states)
                for j in range(num_states)
            )
            for i in range(num_states):
                gamma[t][i] = sum(
                    alpha[t][i] * A[i][j] * B[j][observations[t + 1]] * beta[t + 1][j]
                    for j in range(num_states)
                ) / denom
                for j in range(num_states):
                    xi[t][i][j] = (
                        alpha[t][i] * A[i][j] * B[j][observations[t + 1]] * beta[t + 1][j]
                    ) / denom

        denom = sum(alpha[T - 1][i] for i in range(num_states))
        for i in range(num_states):
            gamma[T - 1][i] = alpha[T - 1][i] / denom

        # M-step: Update A, B, and pi
        for i in range(num_states):
            for j in range(num_states):
                numer = sum(xi[t][i][j] for t in range(T - 1))
                denom = sum(gamma[t][i] for t in range(T - 1))
                A[i][j] = numer / denom if denom != 0 else 0

        for i in range(num_states):
            for k in range(num_emissions):
                numer = sum(
                    gamma[t][i] for t in range(T) if observations[t] == k
                )
                denom = sum(gamma[t][i] for t in range(T))
                B[i][k] = numer / denom if denom != 0 else 0

        pi[0] = gamma[0]

        # Check convergence
        if abs(log_likelihood - previous_log_likelihood) < tolerance:
            break
        previous_log_likelihood = log_likelihood

    return A, B

def format_output(matrix):
    """Formats the matrix output as required."""
    rows, cols = len(matrix), len(matrix[0])
    output = f"{rows} {cols} " + " ".join(f"{x:.6f}" for row in matrix for x in row)
    return output

def main():
    # Read inputs
    input_data = sys.stdin.read().strip().split('\n')
    A = read_matrix(input_data[0])  # Transition matrix
    B = read_matrix(input_data[1])  # Emission matrix
    pi = read_matrix(input_data[2])  # Initial state probabilities
    observations = read_observations(input_data[3])  # Observation sequence

    # Train the model using Baum-Welch
    A_est, B_est = baum_welch(A, B, pi, observations)

    # Print results
    print(format_output(A_est))
    print(format_output(B_est))

if __name__ == "__main__":
    main()
