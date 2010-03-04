This generator permits to generate a Complex Hermitian Positive Definite matrix 
in order to execute a Cholesky decomposition on it. 

In this directory there are two files:

- Hermitian.m
- CholeskyGenerator.m

The generator can be executed on Matlab or on Octave / QTOctave.

To generate the matrix follow these simple istructions:

- Start matlab or octave
- Open the directory in with the utility is placed
- Execute the following function
  	  CholeskyGenerator(MATRIX_SIZE, 'MATRIX_FILENAME', 'MATRIX_TRANSPOSED_FILENAME', 'CHOLESKY_RESULT_FILENAME');

where the arguments are as follow:

- MATRIX_SIZE: size of the matrix to generate
- MATRIX_FILENAME: file where the matrix is saved
- MATRIX_TRANSPOSED_FILENAME: file where the matrix is saved in transposed form, so that you can execute the
  "transposed" version of the Cholesky Decomposition benchmark.
- CHOLESKY_RESULT_FILENAME: file where the factorized matrix is saved, you can use it with the "correctnessCheck" utility



