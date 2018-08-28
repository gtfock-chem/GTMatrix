LIB     = ../libBuzzMatrix.a
MPICC   = mpiicc
CFLAGS  = -Wall -g -O3 -qopenmp -std=gnu99
LDFLAGS = -qopenmp
AR      = xiar rcs

EXES = test_Get_Put.x test_1D_partition.x

all: $(EXES)

test_Get_Put.x: test_Get_Put.c
	$(MPICC) ${CFLAGS} -I../ -c test_Get_Put.c -o test_Get_Put.o
	$(MPICC) ${LDFLAGS} -o test_Get_Put.x test_Get_Put.o ${LIB} 

test_1D_partition.x: test_1D_partition.c
	$(MPICC) ${CFLAGS} -I../ -c test_1D_partition.c -o test_1D_partition.o
	$(MPICC) ${LDFLAGS} -o test_1D_partition.x test_1D_partition.o ${LIB} 

clean:
	rm -f $(EXES)
