LIB     = ../libGTMatrix.a
MPICC   ?= mpiicc
CFLAGS  = -Wall -g -O3 -std=gnu99

SRCS = $(wildcard *.c)
OBJS = $(SRCS:.c=.o)
EXES = $(OBJS:.o=.x)

all: $(EXES)

%.o: %.c
	$(MPICC) ${CFLAGS} -I../ -c $^ 
	
%.x: %.o
	$(MPICC) ${LDFLAGS} -o $@ $^ ${LIB} 
	
clean:
	rm -f $(EXES)
