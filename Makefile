# 
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# 

CC = c++
CFLAGS = -pthread -O3 -funroll-loops -std=c++0x
OBJS = args.o dictionary.o matrix.o vector.o model.o utils.o
INCLUDES = -I.

all: fasttext 

args.o: src/args.cc src/args.h
	$(CC) $(CFLAGS) -c src/args.cc

dictionary.o: src/dictionary.cc src/dictionary.h src/args.h
	$(CC) $(CFLAGS) -c src/dictionary.cc

matrix.o: src/matrix.cc src/matrix.h src/utils.h
	$(CC) $(CFLAGS) -c src/matrix.cc

vector.o: src/vector.cc src/vector.h src/utils.h
	$(CC) $(CFLAGS) -c src/vector.cc

model.o: src/model.cc src/model.h src/args.h
	$(CC) $(CFLAGS) -c src/model.cc

utils.o: src/utils.cc src/utils.h
	$(CC) $(CFLAGS) -c src/utils.cc

fasttext : $(OBJS) src/fasttext.cc
	$(CC) $(CFLAGS) $(OBJS) src/fasttext.cc -o fasttext

clean:
	rm -rf *.o fasttext 
