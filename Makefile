LIB     = libBuzzMatrix.a
MPICC   = mpiicc
CFLAGS  = -Wall -g -O3 -qopenmp -std=gnu99
LDFLAGS = -qopenmp
AR      = xiar rcs

OBJS = Buzz_Matrix_Typedef.o Buzz_Req_Vector.o   Buzz_Matrix_Get.o  \
       Buzz_Matrix_Update.o  Buzz_Matrix_Other.o Buzz_Task_Queue.o utils.o 

$(LIB): $(OBJS) 
	${AR} $@ $^
	
Buzz_Matrix_Typedef.o: Makefile Buzz_Matrix_Typedef.h Buzz_Matrix_Typedef.c 
	$(MPICC) ${CFLAGS} -c Buzz_Matrix_Typedef.c -o $@ 
	
Buzz_Req_Vector.o: Makefile Buzz_Req_Vector.h Buzz_Req_Vector.c 
	$(MPICC) ${CFLAGS} -c Buzz_Req_Vector.c -o $@ 
	
Buzz_Matrix_Get.o: Makefile Buzz_Matrix_Typedef.h utils.h Buzz_Matrix_Get.c
	$(MPICC) ${CFLAGS} -c Buzz_Matrix_Get.c -o $@ 

Buzz_Matrix_Update.o: Makefile Buzz_Matrix_Typedef.h utils.h  Buzz_Matrix_Update.c
	$(MPICC) ${CFLAGS} -c Buzz_Matrix_Update.c -o $@ 

Buzz_Matrix_Other.o: Makefile Buzz_Matrix_Typedef.h Buzz_Matrix_Get.h Buzz_Matrix_Other.c 
	$(MPICC) ${CFLAGS} -c Buzz_Matrix_Other.c -o $@ 
	
Buzz_Task_Queue.o: Makefile Buzz_Task_Queue.h Buzz_Task_Queue.c
	$(MPICC) ${CFLAGS} -c Buzz_Task_Queue.c -o $@ 

utils.o: Makefile utils.c utils.h
	$(MPICC) ${CFLAGS} -c utils.c -o $@ 

clean:
	rm -f $(OBJS) $(LIB)
