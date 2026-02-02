In this task, I calculate the most likely sequence of (hidden) states that the system moves through given an emission sequence and an HMM model.

Input
Three matrices; transition matrix, emission matrix, and initial state probability distribution followed by the number of emissions and the sequence of emissions itself. The initial state probability distribution is a row vector encoded as a matrix with only one row. Each matrix is given on a separate line with the number of rows and columns followed by the matrix elements (ordered row by row). Note that the rows and column size can be different from the sample input. It is assumed that there are M different discrete emission types and these are indexed 0 through M-1 in the emission sequence. For example, if there were M=3 possible different emissions (could be the three colours red, green and blue for example), they would be identified by 0, 1 and 2 in the emission sequence.

Output
The most probable sequence of states as zero-based indices separated by spaces.
