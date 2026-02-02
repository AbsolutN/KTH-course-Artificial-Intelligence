def parse_matrix(input_line):
    """
    Parses a line of input into a 2D matrix.
    """
    elements = list(map(float, input_line.split()))
    rows = int(elements.pop(0))
    cols = int(elements.pop(0))
    return [elements[i * cols:(i + 1) * cols] for i in range(rows)]


def parse_observations(input_line):
    """
    Parses the observation sequence from input.
    """
    elements = list(map(int, input_line.split()))
    return elements[1:]  # Skip the count of observations


def forward_algorithm(transition_matrix, emission_matrix, initial_probabilities, observations):
    """
    Implements the forward algorithm for Hidden Markov Models.
    """
    num_states = len(initial_probabilities[0])
    alpha = [[0] * num_states for _ in range(len(observations))]

    # Initialization step
    for state in range(num_states):
        alpha[0][state] = initial_probabilities[0][state] * emission_matrix[state][observations[0]]

    # Recursive computation
    for t in range(1, len(observations)):
        for state in range(num_states):
            alpha[t][state] = sum(
                alpha[t - 1][prev_state] * transition_matrix[prev_state][state]
                for prev_state in range(num_states)
            ) * emission_matrix[state][observations[t]]

    # Termination step
    return sum(alpha[-1])


def main():
    import sys
    input_data = sys.stdin.read().strip().split("\n")

    # Parse input
    transition_matrix = parse_matrix(input_data[0])
    emission_matrix = parse_matrix(input_data[1])
    initial_probabilities = parse_matrix(input_data[2])
    observations = parse_observations(input_data[3])

    # Perform forward algorithm
    result = forward_algorithm(transition_matrix, emission_matrix, initial_probabilities, observations)

    # Print the final result rounded to six decimal places
    print(f"{result:.6f}")


if __name__ == "__main__":
    main()
