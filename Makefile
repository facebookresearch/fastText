#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
NVCC = nvcc --default-stream per-thread
NVCCFLAGS = --compiler-options -pthread -std=c++11
CXX = c++
CXXFLAGS = -pthread -std=c++0x -march=native
OBJS = args.o matrix.o dictionary.o loss.o productquantizer.o densematrix.o quantmatrix.o vector.o model.o utils.o meter.o fasttext.o
INCLUDES = -I. -I/usr/local/cuda/include/

opt: CXXFLAGS += -O3 -funroll-loops
opt: fasttext

coverage: CXXFLAGS += -O0 -fno-inline -fprofile-arcs --coverage
coverage: fasttext

debug: CXXFLAGS += -g -O0 -fno-inline
debug: fasttext

cudadebug: NVCCFLAGS = -DFASTTEXT_CUDA_DEBUG --compiler-options -pthread -std=c++11 -G -g
cudadebug: CXXFLAGS += -DFASTTEXT_CUDA -DFASTTEXT_CUDA_DEBUG -g -O0 -fno-inline
cudadebug: fasttext_cuda

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

utils.o: src/utils.cc src/utils.h
	$(CXX) $(CXXFLAGS) -c src/utils.cc

meter.o: src/meter.cc src/meter.h
	$(CXX) $(CXXFLAGS) -c src/meter.cc

fasttext.o: src/fasttext.cc src/*.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/fasttext.cc

fasttext: $(OBJS) src/fasttext.cc
	$(CXX) $(CXXFLAGS) $(OBJS) src/main.cc -o fasttext

cudaloss.o: src/cudaloss.cu src/*.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c src/cudaloss.cu

fasttext_cuda: CXXFLAGS += -O3 -funroll-loops -fno-inline -DFASTTEXT_CUDA
fasttext_cuda: $(OBJS) src/fasttext.cc cudaloss.o
	$(NVCC) $(NVCCFLAGS) $(OBJS) cudaloss.o -lcudnn -lcublas src/main.cc -o fasttext_cuda

clean:
	rm -rf *.o *.gcno *.gcda fasttext fasttext_cuda
