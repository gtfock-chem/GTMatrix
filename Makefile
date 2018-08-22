LIB     = libBuzzMatrix.a
MPICC   = mpiicc
CFLAGS  = -Wall -g -O3 -qopenmp -std=gnu99 -xHost
LDFLAGS = -qopenmp
AR      = xiar rcs

OBJS = utils.o Buzz_Matrix.o

$(LIB): $(OBJS) 
	${AR} $@ $^

utils.o: Makefile utils.c utils.h
	$(MPICC) ${CFLAGS} -c utils.c       -o $@ 
	
Buzz_Matrix.o: Makefile Buzz_Matrix.c Buzz_Matrix.h
	$(MPICC) ${CFLAGS} -c Buzz_Matrix.c -o $@ 

test: Makefile $(OBJS) 
	$(MPICC) $(CFLAGS) $(LDFLAGS) $(OBJS) test_Buzz_Matrix.c -o test_Buzz_Matrix.x

clean:
	rm -f $(OBJS) $(LIB) test_Buzz_Matrix.x
