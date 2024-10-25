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
set(BMCU_FLAGS -Wall -Wextra -Wsign-conversion -Wfloat-conversion)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    list(APPEND BMCU_FLAGS -O3 -DNDEBUG -march=native -mtune=native)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND BMCU_FLAGS -O0 -g)
endif()

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test")

enable_testing()

add_executable(bf-serial "${SRC_DIR}/bf-serial.c")
target_compile_options(bf-serial PRIVATE ${BMCU_FLAGS} -Wpedantic -Werror)
add_executable(graphgen "${SRC_DIR}/graphgen.c")
target_compile_options(graphgen PRIVATE ${BMCU_FLAGS} -Wpedantic -Werror)

foreach(version 0-mutex 0-none 1 2)
    foreach(structure aos soa)
        foreach(shmem sh nosh)
            set(executable_name "bf${version}-${structure}-${shmem}")
            add_executable(${executable_name} "${SRC_DIR}/${executable_name}.cu")
            target_compile_options(${executable_name} PRIVATE ${BMCU_FLAGS} -Werror all-warnings)

            foreach(test 025 050 075 100 DE ME NV VT rome)
                set(input_file "${TEST_DIR}/graph-${test}.in")
                set(solution_file "${TEST_DIR}/graph-${test}.sol")
                add_test(NAME ${executable_name}_${test} COMMAND ${executable_name} ${input_file} ${solution_file})
            endforeach()
        endforeach()
    endforeach()
endforeach()

# Convert BMCU_FLAGS to string before printing it
string(REPLACE ";" " " BMCU_FLAGS "${BMCU_FLAGS}")

message(STATUS "")
message(STATUS "Successfully configured ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "CMAKE_C_COMPILER         : ${CMAKE_C_COMPILER} v${CMAKE_C_COMPILER_VERSION}")
message(STATUS "CMAKE_CUDA_COMPILER      : ${CMAKE_CUDA_COMPILER} v${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CMAKE_CUDA_ARCHITECTURES : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CMAKE_BUILD_TYPE         : ${CMAKE_BUILD_TYPE}")
message(STATUS "BMCU_FLAGS               : ${BMCU_FLAGS}")
message(STATUS "")
