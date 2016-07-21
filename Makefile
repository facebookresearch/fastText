CC = cc
CFLAGS = -lm -pthread -O3 -funroll-loops -std=c++0x
OBJS = Args.o Dictionary.o Matrix.o Vector.o Model.o Utils.o
INCLUDES = -I.

all: fasttext print-vectors classify

Dictionary.h: Real.h
Matrix.h: Real.h Vector.h
Vector.h: Real.h Matrix.h
Model.h: Vector.h Matrix.h Real.h
Utils.h: Real.h

Args.o: Args.cpp Args.h
	$(CC) $(CFLAGS) -c Args.cpp

Dictionary.o: Dictionary.cpp Dictionary.h Args.h
	$(CC) $(CFLAGS) -c Dictionary.cpp

Matrix.o: Matrix.cpp Matrix.h Utils.h
	$(CC) $(CFLAGS) -c Matrix.cpp

Vector.o: Vector.cpp Vector.h Utils.h
	$(CC) $(CFLAGS) -c Vector.cpp

Model.o: Model.cpp Model.h Args.h
	$(CC) $(CFLAGS) -c Model.cpp

Utils.o: Utils.cpp Utils.h
	$(CC) $(CFLAGS) -c Utils.cpp

fasttext : $(OBJS) FastText.cpp
	$(CC) $(CFLAGS) $(OBJS) FastText.cpp -o fasttext
print-vectors : $(OBJS) PrintVectors.cpp
	$(CC) $(CFLAGS) $(OBJS) PrintVectors.cpp -o print-vectors
classify : $(OBJS) Classify.cpp
	$(CC) $(CFLAGS) $(OBJS) Classify.cpp -o classify

clean:
	rm -rf *.o fasttext print-vectors classify
