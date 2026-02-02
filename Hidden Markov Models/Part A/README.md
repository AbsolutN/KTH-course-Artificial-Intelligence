_In_ this task I predict how the system will evolve over time and estimate the probability for different emissions / events in the system i.e. what can be observed from the HMM. I was given the state probability distribution (i.e. the probability that the system is in each of the N states), the transition matrix (i.e. the matrix that gives the probability to transition from one state to another) and the emission matrix (i.e. the matrix that gives the probability for the different emissions / events / observations given a certain state).

Input
Three matrices (in this order); transition matrix, emission matrix, and initial state probability distribution. The initial state probability distribution is a row vector encoded as a matrix with only one row. Each matrix is given on a separate line with the number of rows and columns followed by the matrix elements (ordered row by row). Note that the rows and column size can be different from the sample input.

Output
The emission probability distribution on a single line in the same matrix format, including the dimensions.
