import sys

def read_matrix(input_line):
    """Reads a matrix from a string input."""
    data = list(map(float, input_line.split()))
    rows, cols = int(data[0]), int(data[1])
    matrix = []
    idx = 2  # Start after the size specifier
    for _ in range(rows):
        matrix.append(data[idx:idx + cols])
        idx += cols
    return matrix

def read_observations(input_line):
    """Reads the sequence of observations from a string input."""
    data = list(map(int, input_line.split()))
    return data[1:]  # Ignore the first element, which is the count of observations

def viterbi_algorithm(transition_matrix, emission_matrix, initial_distribution, observations):
    """
    Implements the Viterbi algorithm for Hidden Markov Models.

    Args:
        transition_matrix: State transition probability matrix.
        emission_matrix: Emission probability matrix.
        initial_distribution: Initial state probabilities.
        observations: Sequence of observed emissions.

    Returns:
        The most probable sequence of states.
    """
    num_states = len(initial_distribution[0])
    T = len(observations)
    
    # Initialize the Viterbi table and path tracker
    dp = [[0] * num_states for _ in range(T)]
    backtrack = [[0] * num_states for _ in range(T)]
    
    # Initialization step
    for state in range(num_states):
        dp[0][state] = initial_distribution[0][state] * emission_matrix[state][observations[0]]
        backtrack[0][state] = 0

    # Recursive computation
    for t in range(1, T):
        for state in range(num_states):
            max_prob, max_state = max(
                (dp[t - 1][prev_state] * transition_matrix[prev_state][state], prev_state)
                for prev_state in range(num_states)
            )
            dp[t][state] = max_prob * emission_matrix[state][observations[t]]
            backtrack[t][state] = max_state

    # Traceback step
    most_probable_sequence = [0] * T
    most_probable_sequence[-1] = max(range(num_states), key=lambda state: dp[-1][state])

    for t in range(T - 2, -1, -1):
        most_probable_sequence[t] = backtrack[t + 1][most_probable_sequence[t + 1]]

    return most_probable_sequence

def main():
    # Read input data from stdin
    input_data = sys.stdin.read().strip().split('\n')
    
    transition_matrix = read_matrix(input_data[0])  # Transition matrix (A)
    emission_matrix = read_matrix(input_data[1])  # Emission matrix (B)
    initial_distribution = read_matrix(input_data[2])  # Initial probabilities (pi)
    observations = read_observations(input_data[3])  # Sequence of observations

    # Calculate the most probable state sequence using the Viterbi algorithm
    most_probable_sequence = viterbi_algorithm(transition_matrix, emission_matrix, initial_distribution, observations)
    print(" ".join(map(str, most_probable_sequence)))


if __name__ == "__main__":
    main()
