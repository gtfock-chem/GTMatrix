LIB     = libBuzzMatrix.a
MPICC   = mpiicc
CFLAGS  = -Wall -g -O3 -qopenmp -std=gnu99
LDFLAGS = -qopenmp
AR      = xiar rcs

OBJS = utils.o Buzz_Matrix.o Buzz_Matrix_Get.o Buzz_Matrix_Update.o Buzz_Req_Vector.o

$(LIB): $(OBJS) 
	${AR} $@ $^

utils.o: Makefile utils.c utils.h
	$(MPICC) ${CFLAGS} -c utils.c -o $@ 
	
Buzz_Matrix.o: Makefile Buzz_Matrix.c Buzz_Matrix.h utils.h
	$(MPICC) ${CFLAGS} -c Buzz_Matrix.c -o $@ 

Buzz_Matrix_Get.o: Makefile Buzz_Matrix_Get.c Buzz_Matrix.h utils.h
	$(MPICC) ${CFLAGS} -c Buzz_Matrix_Get.c -o $@ 

Buzz_Matrix_Update.o: Makefile Buzz_Matrix_Update.c Buzz_Matrix.h utils.h
	$(MPICC) ${CFLAGS} -c Buzz_Matrix_Update.c -o $@ 

Buzz_Req_Vector.o: Makefile Buzz_Req_Vector.h
	$(MPICC) ${CFLAGS} -c Buzz_Req_Vector.c -o $@ 
	
clean:
	rm -f $(OBJS) $(LIB)
