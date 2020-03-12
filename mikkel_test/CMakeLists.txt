cmake_minimum_required(VERSION 3.10)

project(xor)
add_executable(xor test.cpp CSVreader.cpp)
include_directories(${PROJECT_SOURCE_DIR})
include_directories(SYSTEM ${NNPACK_INCLUDE_DIR})
include_directories(.)
find_package(Threads REQUIRED)
target_link_libraries(xor Threads::Threads)




