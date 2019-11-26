cmake_minimum_required(VERSION 3.10)

set(ProjectName "main")

set(CMAKE_CXX_FLAGS "-Wall -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Debug)" FORCE)
endif()

project(${ProjectName})
add_executable(${ProjectName} main.cpp)

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

target_include_directories(${ProjectName} PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(${ProjectName}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
    nvonnxparser
    nvinfer_plugin
    cudnn
)

file(COPY mobilenetv2-1.0.onnx DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY synset.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)


