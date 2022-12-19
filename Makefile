#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

CXX = c++
CXXFLAGS = -std=c++11 -march=native
OBJS = args.o matrix.o dictionary.o loss.o productquantizer.o densematrix.o quantmatrix.o vector.o model.o fasttext.o
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

#######
# from https://github.com/lxml/lxml/blob/2cd510258d03887dfad69e77edc47f8bf28773ae/Makefile

FTFVERSION:=$(shell python setup.py -V' )
PYTHON_BUILD_VERSION ?= *
MANYLINUX_CFLAGS=-O3 -g1 -pipe -fPIC -flto
MANYLINUX_LDFLAGS=-flto

.PHONY: sdist build require-cython wheel_manylinux wheel

dist/fasttext-predict-$(FTFVERSION).tar.gz:
	python setup.py sdist

sdist: dist/fasttext-predict-$(FTFVERSION).tar.gz

qemu-user-static:
	docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

wheel_%: dist/fasttext-predict-$(FTFVERSION).tar.gz qemu-user-static
	time docker run --rm -t \
		-v $(shell pwd):/io \
		-e AR=gcc-ar \
		-e NM=gcc-nm \
		-e RANLIB=gcc-ranlib \
		-e CFLAGS="$(MANYLINUX_CFLAGS) $(if $(patsubst %aarch64,,$@),-march=core2,-march=armv8-a -mtune=cortex-a72)" \
		-e LDFLAGS="$(MANYLINUX_LDFLAGS)" \
		-e PYTHON_BUILD_VERSION="$(PYTHON_BUILD_VERSION)" \
		-e WHEELHOUSE=$(subst wheel_,wheelhouse/,$@) \
		quay.io/pypa/$(subst wheel_,,$@) \
		bash /io/build-wheels.sh /io/$<

wheel:
	python setup.py bdist_wheel
