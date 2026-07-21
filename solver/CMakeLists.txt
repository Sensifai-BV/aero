cmake_minimum_required(VERSION 3.16)
project(aero_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ONNX Runtime location. Set -DORT_ROOT=/path/to/onnxruntime (the dir that
# contains include/ and lib/). Falls back to common system prefixes.
set(ORT_ROOT "" CACHE PATH "ONNX Runtime install root (has include/ and lib/)")

find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ORT_ROOT}/include ${ORT_ROOT}/include/onnxruntime
        /usr/include /usr/local/include
        /usr/include/onnxruntime /usr/local/include/onnxruntime)

find_library(ORT_LIB onnxruntime
  HINTS ${ORT_ROOT}/lib /usr/lib /usr/local/lib
        /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)

if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIB)
  message(FATAL_ERROR
    "ONNX Runtime not found. Pass -DORT_ROOT=/path/to/onnxruntime "
    "(the directory containing include/ and lib/). "
    "include=${ORT_INCLUDE_DIR} lib=${ORT_LIB}")
endif()

message(STATUS "ORT include: ${ORT_INCLUDE_DIR}")
message(STATUS "ORT lib    : ${ORT_LIB}")

add_executable(aero_infer aero_infer.cpp)
target_include_directories(aero_infer PRIVATE ${ORT_INCLUDE_DIR})
target_link_libraries(aero_infer PRIVATE ${ORT_LIB})
