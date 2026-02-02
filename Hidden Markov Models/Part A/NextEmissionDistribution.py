#!/usr/bin/env python3
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

def multiply_matrices(A, B):
    """Multiplies two matrices A and B."""
    return [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]

def main():
    # Read inputs
    input_data = sys.stdin.read().strip().split('\n')
    A = read_matrix(input_data[0])  # Transition matrix
    B = read_matrix(input_data[1])  # Emission matrix
    pi = read_matrix(input_data[2])  # Initial state probability distribution

    # Perform the calculations: pi * A -> intermediate, then intermediate * B
    intermediate = multiply_matrices(pi, A)
    emission_probabilities = multiply_matrices(intermediate, B)

    # Format output
    rows, cols = len(emission_probabilities), len(emission_probabilities[0])
    output = f"{rows} {cols} " + " ".join(f"{x:.6g}" for row in emission_probabilities for x in row)
    print(output)

if __name__ == "__main__":
    main()
