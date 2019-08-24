LIB     = ../libGTMatrix.a
MPICC   = mpiicc
CFLAGS  = -Wall -g -O3 -qopenmp -std=c99
LDFLAGS = -qopenmp
AR      = xiar rcs

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
