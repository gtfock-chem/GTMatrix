mpiicpc Buzz_Matrix.c      -c -o Buzz_Matrix.o
mpiicpc test_Buzz_Matrix.c -c -o test_Buzz_Matrix.o
mpiicpc utils.c -c -o utils.o
mpiicpc Buzz_Matrix.o test_Buzz_Matrix.o utils.o -o test_Buzz_Matrix.x