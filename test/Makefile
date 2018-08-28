LIB     = ../libBuzzMatrix.a
MPICC   = mpiicc
CFLAGS  = -Wall -g -O3 -qopenmp -std=gnu99
LDFLAGS = -qopenmp
AR      = xiar rcs

EXES = test_Get_Put.x

test_Get_Put.x: test_Get_Put.c
	$(MPICC) ${CFLAGS} -I../ -c test_Get_Put.c -o test_Get_Put.o
	$(MPICC) ${LDFLAGS} -o test_Get_Put.x test_Get_Put.o ${LIB} 
	
all: $(EXES)

clean:
	rm -f $(EXES)
