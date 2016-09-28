cmake_minimum_required(VERSION 3.6)
project(fastText)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -funroll-loops")

set(SOURCE_FILES
    src/args.cc
    src/args.h
    src/dictionary.cc
    src/dictionary.h
    src/fasttext.cc
    src/matrix.cc
    src/matrix.h
    src/model.cc
    src/model.h
    src/real.h
    src/utils.cc
    src/utils.h
    src/vector.cc
    src/vector.h)

add_library(fasttext ${SOURCE_FILES})
add_executable(fastText src/fasttext.cc)
target_link_libraries(fastText pthread fasttext)