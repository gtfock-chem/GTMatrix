LIB     = libGTMatrix.a
MPICC   = mpiicc
CFLAGS  = -Wall -Wunused-variable -g -O3 -qopenmp -std=c99 
LDFLAGS = -qopenmp
AR      = xiar rcs

OBJS = GTMatrix_Typedef.o GTMatrix_Get.o GTMatrix_Update.o      \
       GTMatrix_Other.o GTM_Req_Vector.o GTM_Task_Queue.o utils.o 

$(LIB): $(OBJS) 
	${AR} $@ $^
	
GTMatrix_Typedef.o: Makefile GTMatrix_Typedef.h GTMatrix_Typedef.c 
	$(MPICC) ${CFLAGS} -c GTMatrix_Typedef.c -o $@ 
	
GTM_Req_Vector.o: Makefile GTM_Req_Vector.h GTM_Req_Vector.c 
	$(MPICC) ${CFLAGS} -c GTM_Req_Vector.c -o $@ 
	
GTMatrix_Get.o: Makefile GTMatrix_Typedef.h utils.h GTMatrix_Get.c
	$(MPICC) ${CFLAGS} -c GTMatrix_Get.c -o $@ 

GTMatrix_Update.o: Makefile GTMatrix_Typedef.h utils.h  GTMatrix_Update.c
	$(MPICC) ${CFLAGS} -c GTMatrix_Update.c -o $@ 

GTMatrix_Other.o: Makefile GTMatrix_Typedef.h GTMatrix_Get.h GTMatrix_Other.c 
	$(MPICC) ${CFLAGS} -c GTMatrix_Other.c -o $@ 
	
GTM_Task_Queue.o: Makefile GTM_Task_Queue.h GTM_Task_Queue.c
	$(MPICC) ${CFLAGS} -c GTM_Task_Queue.c -o $@ 

utils.o: Makefile utils.c utils.h
	$(MPICC) ${CFLAGS} -c utils.c -o $@ 

clean:
	rm -f $(OBJS) $(LIB)
