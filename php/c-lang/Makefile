EXEC = sample
CC = gcc
LDFLAGS = -pthread -lm -lstdc++
COMMON = -I../include -I../../src
CFLAGS = -Wall -Wno-unknown-pragmas -Wfatal-errors -O0 -g
#CFLAGS = -Wall -O0 -g -msse3 -funroll-loops -Wno-unused-parameter -Wno-unknown-pragmas
#CFLAGS = -Wall -O3 -msse3 -funroll-loops -Wno-unused-parameter -Wno-unknown-pragmas
OBJS = main.o

all: main.o
	$(CC) $(COMMON) $(CFLAGS) $(OBJS) ../libfasttext2.a -o $(EXEC) $(LDFLAGS)

main.o: main.c
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

clean:
	rm -rf ./*.o $(EXEC)


#gcc -Iinclude/  obj/captcha.o libdarknet.a -o darknet -pthread -lstdc++  libdarknet.a
