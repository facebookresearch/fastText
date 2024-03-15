#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

CXX = c++
CXXFLAGS = -pthread -std=c++17 -march=native
OBJS = args.o autotune.o matrix.o dictionary.o loss.o productquantizer.o densematrix.o quantmatrix.o vector.o model.o utils.o meter.o fasttext.o
INCLUDES = -I.

opt: CXXFLAGS += -O3 -funroll-loops -DNDEBUG
opt: fasttext

coverage: CXXFLAGS += -O0 -fno-inline -fprofile-arcs --coverage
coverage: fasttext

debug: CXXFLAGS += -g -O0 -fno-inline
debug: fasttext

wasm: webassembly/fasttext_wasm.js

wasmdebug: export EMCC_DEBUG=1
wasmdebug: webassembly/fasttext_wasm.js


args.o: src/args.cc src/args.h
	$(CXX) $(CXXFLAGS) -c src/args.cc

autotune.o: src/autotune.cc src/autotune.h
	$(CXX) $(CXXFLAGS) -c src/autotune.cc

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
	$(CXX) $(CXXFLAGS) -c src/fasttext.cc

fasttext: $(OBJS) src/fasttext.cc src/main.cc
	$(CXX) $(CXXFLAGS) $(OBJS) src/main.cc -o fasttext

clean:
	rm -rf *.o *.gcno *.gcda fasttext *.bc webassembly/fasttext_wasm.js webassembly/fasttext_wasm.wasm


EMCXX = em++
EMCXXFLAGS = --bind --std=c++11 -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s "EXTRA_EXPORTED_RUNTIME_METHODS=['addOnPostRun', 'FS']" -s "DISABLE_EXCEPTION_CATCHING=0" -s "EXCEPTION_DEBUG=1" -s "FORCE_FILESYSTEM=1" -s "MODULARIZE=1" -s "EXPORT_ES6=1" -s 'EXPORT_NAME="FastTextModule"' -Isrc/
EMOBJS = args.bc autotune.bc matrix.bc dictionary.bc loss.bc productquantizer.bc densematrix.bc quantmatrix.bc vector.bc model.bc utils.bc meter.bc fasttext.bc main.bc


main.bc: webassembly/fasttext_wasm.cc
	$(EMCXX) $(EMCXXFLAGS)  webassembly/fasttext_wasm.cc -o main.bc

args.bc: src/args.cc src/args.h
	$(EMCXX) $(EMCXXFLAGS)  src/args.cc -o args.bc

autotune.bc: src/autotune.cc src/autotune.h
	$(EMCXX) $(EMCXXFLAGS)  src/autotune.cc -o autotune.bc

matrix.bc: src/matrix.cc src/matrix.h
	$(EMCXX) $(EMCXXFLAGS) src/matrix.cc -o matrix.bc

dictionary.bc: src/dictionary.cc src/dictionary.h src/args.h
	$(EMCXX) $(EMCXXFLAGS)  src/dictionary.cc -o dictionary.bc

loss.bc: src/loss.cc src/loss.h src/matrix.h src/real.h
	$(EMCXX) $(EMCXXFLAGS) src/loss.cc -o loss.bc

productquantizer.bc: src/productquantizer.cc src/productquantizer.h src/utils.h
	$(EMCXX) $(EMCXXFLAGS)  src/productquantizer.cc -o productquantizer.bc

densematrix.bc: src/densematrix.cc src/densematrix.h src/utils.h src/matrix.h
	$(EMCXX) $(EMCXXFLAGS) src/densematrix.cc -o densematrix.bc

quantmatrix.bc: src/quantmatrix.cc src/quantmatrix.h src/utils.h src/matrix.h
	$(EMCXX) $(EMCXXFLAGS) src/quantmatrix.cc -o quantmatrix.bc

vector.bc: src/vector.cc src/vector.h src/utils.h
	$(EMCXX) $(EMCXXFLAGS)  src/vector.cc -o vector.bc

model.bc: src/model.cc src/model.h src/args.h
	$(EMCXX) $(EMCXXFLAGS)  src/model.cc -o model.bc

utils.bc: src/utils.cc src/utils.h
	$(EMCXX) $(EMCXXFLAGS)  src/utils.cc -o utils.bc

meter.bc: src/meter.cc src/meter.h
	$(EMCXX) $(EMCXXFLAGS)  src/meter.cc -o meter.bc

fasttext.bc: src/fasttext.cc src/*.h
	$(EMCXX) $(EMCXXFLAGS)  src/fasttext.cc -o fasttext.bc

webassembly/fasttext_wasm.js: $(EMOBJS) webassembly/fasttext_wasm.cc Makefile
	$(EMCXX) $(EMCXXFLAGS) $(EMOBJS) -o webassembly/fasttext_wasm.js


