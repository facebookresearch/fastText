#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

CXX=c++

CXXFLAGS=-std=c++0x -pthread

opt: CXXFLAGS+=-O3 -funroll-loops
opt: lib fasttext

debug: CXXFLAGS+=-g -O0 -fno-inline
debug: lib fasttext

INCLUDES=-Isrc/

SOURCES=${wildcard src/*.cc}

OBJECTS=${SOURCES:.cc=.o}

LIB=src/libfasttext.a

all: lib fasttext

lib: ${OBJECTS}
	ar -cq ${LIB} ${OBJECTS}
	ranlib ${LIB}

fasttext: src/fasttext.cc ${OBJECTS}
	${CXX} ${CXXFLAGS} ${INCLUDES} -o fasttext src/fasttext.cc ${LIB}

%.o: %.cc
	${CXX} ${CXXFLAGS} ${INCLUDES} -c $< -o $@

clean:
	${RM} ${LIB} ${OBJECTS} fasttext
