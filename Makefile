#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

CXX = c++
CXXFLAGS = -std=c++11 -march=native
OBJS = args.o matrix.o dictionary.o loss.o productquantizer.o densematrix.o quantmatrix.o vector.o model.o utils.o fasttext.o
INCLUDES = -I.

args.o: src/args.cc src/args.h
	$(CXX) $(CXXFLAGS) -c src/args.cc

matrix.o: src/matrix.cc src/matrix.h
	$(CXX) $(CXXFLAGS) -c src/matrix.cc

dictionary.o: src/dictionary.cc src/dictionary.h src/args.h
	$(CXX) $(CXXFLAGS) -c src/dictionary.cc

loss.o: src/loss.cc src/loss.h src/matrix.h src/real.h
	$(CXX) $(CXXFLAGS) -c src/loss.cc

productquantizer.o: src/productquantizer.cc src/productquantizer.h src/utils.h
	$(CXX) $(CXXFLAGS) -c src/productquantizer.cc

densematrix.o: src/densematrix.cc src/densematrix.h src/utils.h src/matrix.h
	$(CXX) $(CXXFLAGS) -c src/densematrix.cc

quantmatrix.o: src/quantmatrix.cc src/quantmatrix.h src/utils.h src/matrix.h
	$(CXX) $(CXXFLAGS) -c src/quantmatrix.cc

vector.o: src/vector.cc src/vector.h src/utils.h
	$(CXX) $(CXXFLAGS) -c src/vector.cc

model.o: src/model.cc src/model.h src/args.h
	$(CXX) $(CXXFLAGS) -c src/model.cc

fasttext.o: src/fasttext.cc src/*.h
	$(CXX) $(CXXFLAGS) -c src/fasttext.cc

clean:
	rm -rf *.o *.gcno *.gcda


