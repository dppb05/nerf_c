CC=gcc
CFLAGS=-Wall -c
EXEDIR=.

all: $(EXEDIR)/nerf

$(EXEDIR)/nerf: stex.o matrix.o util.o nerf.o
	$(CC) stex.o matrix.o util.o nerf.o -o $@ -lm

nerf.o: nerf.c
	$(CC) $(CFLAGS) nerf.c

util.o: util.c
	$(CC) $(CFLAGS) util.c

matrix.o: matrix.c
	$(CC) $(CFLAGS) matrix.c

stex.o: stex.c
	$(CC) $(CFLAGS) stex.c

clean:
	rm -rf *o $(EXEDIR)/nerf
