cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project("Bellman-Ford CUDA" VERSION 1.0)

set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)

if(NOT CMAKE_BUILD_TYPE)
    # Compile in Debug by default
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "The type of build" FORCE)
endif()

enable_language(C)
set(CMAKE_C_STANDARD 99 CACHE STRING "C standard")
set(CMAKE_C_STANDARD_REQUIRED ON)

enable_language(CUDA)

# Default C flags
set(BMCU_FLAGS -Wall -Wextra -Wpedantic -Werror)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    list(APPEND BMCU_FLAGS -O3 -DNDEBUG -march=native -mtune=native)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND BMCU_FLAGS -O0 -g -fsanitize=address,undefined,leak)
endif()

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_executable(bf-serial "${SRC_DIR}/bf-serial.c")
add_executable(graphgen "${SRC_DIR}/graphgen.c")

foreach(version 0-mutex 0-none 1 2)
    foreach(structure aos soa)
        foreach(shmem sh nosh)
            set(name "bf${version}-${structure}-${shmem}")
            add_executable(${name} "${SRC_DIR}/${name}.cu")
        endforeach()
    endforeach()
endforeach()

# Convert BMCU_FLAGS to string before printing it
string(REPLACE ";" " " BMCU_FLAGS "${BMCU_FLAGS}")

message(STATUS "")
message(STATUS "Successfully configured ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "CMAKE_C_COMPILER    : ${CMAKE_C_COMPILER} v${CMAKE_C_COMPILER_VERSION}")
message(STATUS "CMAKE_CUDA_COMPILER : ${CMAKE_CUDA_COMPILER} v${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CMAKE_BUILD_TYPE    : ${CMAKE_BUILD_TYPE}")
message(STATUS "BMCU_FLAGS          : ${BMCU_FLAGS}")
message(STATUS "")
