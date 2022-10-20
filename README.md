# matrics

Do matrix math in shell and terminals

# installation:

matrics requires Fortran and Lapack:

g++ -L/usr/local/gfortran/lib -L/PATH_TO_LAPACK/lib -lgfortran -lm -lblas -llapack -o matrics matrics.cpp

# use:

## matrix math

matrics does matrix math of NxN matrixes in terminal:

matrics [dimension] matrix_A operation [matrix_B] -[format]

matrix_A and matrix_B can be existing files, stdin/con, constant number, or a constant matrix. Files or stdin should contain NxN columns and L lines, where each line represents an NxN matrix.
Dimension is the number of rows/columns of each matrix. The dimension can be missing when it's possible to guess the matrix dimension from matrix_A (e.g. matrix_A is a file or constant matrix.)

For example:

matrics file_A + --"1 2 3 4"    // matrix sum

matrics --"1 2 3 4" . file_B    // matrix product

matrics file_B . 5              // 5 is recognized as 5\times\identity

matrics file_A / file_B  (<-equivalent->)  matrics file_B ^ -1 | matrics file_A . con

matrics file_A // file_B  (<-equivalent->)  matrics file_A ^ -1 | matrics 5 con . file_B

matrics 5 1 exp file_A          // \exp{A}

matrics 5 2 ln file_B           // 2\ln{B}

matrics 100 file_A sub 5        // top 5x5 submatrix of file_A

matrics A ^^ B                  // direct product between A and B

matrics A t/det/tr              // transpose/determinant/trace of A

matrics A ev/vl/vr              // eigenvalues/left-eigenvectors/right-eigenvectors of A

matrics A diag                  // print diagonal matrics of A

matrics A print --"1 3 5 7 .."  // print selected elements of A

matrics A diag-to-rows/diag-to-cols/row1-to-rows/row1-to-diag/col1-to-cols/col1-to-diag

matrics A sum/average/stdev     // compute the sum/average/stdev of each element for the L matrixes of A, L=line_number(A)

## format string

-%.15g       double format, and everything displays in a single line

---%.15g     double format, print in N rows and N columns

-dd          double format, print in N rows and N columns

-wolfram     the matrix format used in Mathematica

## max dimension

The maximum dimension is defined in the environment string $MATRICS_MAXDIM. By default (if this string is not defined) it is 200.
