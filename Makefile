LIB     = libBuzzMatrix.a
MPICC   = mpiicc
CFLAGS  = -Wall -g -O3 -qopenmp -std=gnu99
LDFLAGS = -qopenmp
AR      = xiar rcs

OBJS = utils.o Buzz_Matrix.o Buzz_Matrix_Get.o Buzz_Matrix_Put.o

$(LIB): $(OBJS) 
	${AR} $@ $^

utils.o: Makefile utils.c utils.h
	$(MPICC) ${CFLAGS} -c utils.c           -o $@ 
	
Buzz_Matrix.o: Makefile Buzz_Matrix.c Buzz_Matrix.h utils.h
	$(MPICC) ${CFLAGS} -c Buzz_Matrix.c     -o $@ 

Buzz_Matrix_Get.o: Makefile Buzz_Matrix_Get.c Buzz_Matrix.h utils.h
	$(MPICC) ${CFLAGS} -c Buzz_Matrix_Get.c -o $@ 

Buzz_Matrix_Put.o: Makefile Buzz_Matrix_Put.c Buzz_Matrix.h utils.h
	$(MPICC) ${CFLAGS} -c Buzz_Matrix_Put.c -o $@ 

clean:
	rm -f $(OBJS) $(LIB)
