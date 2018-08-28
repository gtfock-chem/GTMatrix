LIB     = ../libBuzzMatrix.a
MPICC   = mpiicc
CFLAGS  = -Wall -g -O3 -qopenmp -std=gnu99
LDFLAGS = -qopenmp
AR      = xiar rcs

EXES = test_Get_Put.x test_1D_partition.x

all: $(EXES)

test_Get_Put.x: test_Get_Put_Acc.c
	$(MPICC) ${CFLAGS} -I../ -c test_Get_Put_Acc.c -o test_Get_Put_Acc.o
	$(MPICC) ${LDFLAGS} -o test_Get_Put_Acc.x test_Get_Put_Acc.o ${LIB} 

test_1D_partition.x: test_1D_partition.c
	$(MPICC) ${CFLAGS} -I../ -c test_1D_partition.c -o test_1D_partition.o
	$(MPICC) ${LDFLAGS} -o test_1D_partition.x test_1D_partition.o ${LIB} 

clean:
	rm -f $(EXES)
