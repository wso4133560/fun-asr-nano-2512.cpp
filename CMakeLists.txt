cmake_minimum_required(VERSION 3.16)

project(fun_asr_nano LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FUN_ASR_NATIVE_OPT "Enable native CPU tuning flags" ON)
option(FUN_ASR_AUTO_DETECT_CPU "Auto-detect CPU features and set ggml flags" ON)
option(FUN_ASR_AUTO_DETECT_BLAS "Auto-enable ggml BLAS if found" ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Threads REQUIRED)

set(GGML_BUILD_TESTS OFF CACHE BOOL "Disable ggml tests" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "Disable ggml examples" FORCE)

if(FUN_ASR_AUTO_DETECT_CPU AND NOT CMAKE_CROSSCOMPILING)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86")
        include(CheckCXXSourceRuns)

        function(fun_asr_check_cpu_supports feature outvar)
            set(_src "
                #if !(defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
                int main() { return 1; }
                #endif
                #if defined(__GNUC__) || defined(__clang__)
                int main() { return __builtin_cpu_supports(\"${feature}\") ? 0 : 1; }
                #else
                int main() { return 1; }
                #endif
            ")
            check_cxx_source_runs("${_src}" ${outvar})
        endfunction()

        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
            fun_asr_check_cpu_supports("avx"     FUN_ASR_CPU_HAS_AVX)
            fun_asr_check_cpu_supports("avx2"    FUN_ASR_CPU_HAS_AVX2)
            fun_asr_check_cpu_supports("fma"     FUN_ASR_CPU_HAS_FMA)
            fun_asr_check_cpu_supports("f16c"    FUN_ASR_CPU_HAS_F16C)
            fun_asr_check_cpu_supports("bmi2"    FUN_ASR_CPU_HAS_BMI2)
            fun_asr_check_cpu_supports("avx512f" FUN_ASR_CPU_HAS_AVX512)

            set(FUN_ASR_CPU_FEATURES_DETECTED ON)
        endif()
    endif()
endif()

if(FUN_ASR_CPU_FEATURES_DETECTED)
    set(GGML_NATIVE OFF CACHE BOOL "ggml: optimize the build for the current system" FORCE)
    set(GGML_AVX    ${FUN_ASR_CPU_HAS_AVX}    CACHE BOOL "ggml: enable AVX" FORCE)
    set(GGML_AVX2   ${FUN_ASR_CPU_HAS_AVX2}   CACHE BOOL "ggml: enable AVX2" FORCE)
    set(GGML_FMA    ${FUN_ASR_CPU_HAS_FMA}    CACHE BOOL "ggml: enable FMA" FORCE)
    set(GGML_F16C   ${FUN_ASR_CPU_HAS_F16C}   CACHE BOOL "ggml: enable F16C" FORCE)
    set(GGML_BMI2   ${FUN_ASR_CPU_HAS_BMI2}   CACHE BOOL "ggml: enable BMI2" FORCE)
    set(GGML_AVX512 ${FUN_ASR_CPU_HAS_AVX512} CACHE BOOL "ggml: enable AVX512F" FORCE)
endif()

if(FUN_ASR_AUTO_DETECT_BLAS)
    if(NOT DEFINED CACHE{GGML_BLAS})
        find_package(BLAS QUIET)
        if(BLAS_FOUND)
            set(GGML_BLAS ON CACHE BOOL "ggml: use BLAS" FORCE)
        endif()
    endif()
endif()

option(GGML_CUDA "ggml: use CUDA" OFF)

add_subdirectory(ggml)

add_library(fun_asr_lib
    src/fun_asr.cpp)

target_include_directories(fun_asr_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(fun_asr_lib
    PUBLIC
        ggml
        Threads::Threads
        m
        dl)

if(FUN_ASR_NATIVE_OPT AND NOT MSVC)
    target_compile_options(fun_asr_lib PRIVATE -march=native -mtune=native)
endif()

add_executable(fun-asr-nano src/main.cpp)
target_link_libraries(fun-asr-nano PRIVATE fun_asr_lib)
