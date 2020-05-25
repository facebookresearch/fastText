#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

cmake_minimum_required(VERSION 2.8.9)
project(fasttext)

# The version number.
set (fasttext_VERSION_MAJOR 0)
set (fasttext_VERSION_MINOR 1)

include_directories(fasttext)

set(CMAKE_CXX_FLAGS " -pthread -std=c++11 -funroll-loops -O3 -march=native")

set(HEADER_FILES
    src/args.h
    src/autotune.h
    src/densematrix.h
    src/dictionary.h
    src/fasttext.h
    src/loss.h
    src/matrix.h
    src/meter.h
    src/model.h
    src/productquantizer.h
    src/quantmatrix.h
    src/real.h
    src/utils.h
    src/vector.h)

set(SOURCE_FILES
    src/args.cc
    src/autotune.cc
    src/densematrix.cc
    src/dictionary.cc
    src/fasttext.cc
    src/loss.cc
    src/main.cc
    src/matrix.cc
    src/meter.cc
    src/model.cc
    src/productquantizer.cc
    src/quantmatrix.cc
    src/utils.cc
    src/vector.cc)


if (NOT MSVC)
  include(GNUInstallDirs)
  configure_file("fasttext.pc.in" "fasttext.pc" @ONLY)
  install(FILES "${CMAKE_BINARY_DIR}/fasttext.pc" DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()

add_library(fasttext-shared SHARED ${SOURCE_FILES} ${HEADER_FILES})
add_library(fasttext-static STATIC ${SOURCE_FILES} ${HEADER_FILES})
add_library(fasttext-static_pic STATIC ${SOURCE_FILES} ${HEADER_FILES})
set_target_properties(fasttext-shared PROPERTIES OUTPUT_NAME fasttext
  SOVERSION "${fasttext_VERSION_MAJOR}")
set_target_properties(fasttext-static PROPERTIES OUTPUT_NAME fasttext)
set_target_properties(fasttext-static_pic PROPERTIES OUTPUT_NAME fasttext_pic
  POSITION_INDEPENDENT_CODE True)
add_executable(fasttext-bin src/main.cc)
target_link_libraries(fasttext-bin pthread fasttext-static)
set_target_properties(fasttext-bin PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" OUTPUT_NAME fasttext)
install (TARGETS fasttext-shared
    LIBRARY DESTINATION lib)
install (TARGETS fasttext-static
    ARCHIVE DESTINATION lib)
install (TARGETS fasttext-static_pic
    ARCHIVE DESTINATION lib)
install (TARGETS fasttext-bin
    RUNTIME DESTINATION bin
 PUBLIC_HEADER DESTINATION include/fasttext)
