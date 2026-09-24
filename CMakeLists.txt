cmake_minimum_required(VERSION 3.28)

# The current implementation is compiled only for sm_120a. Set and validate the
# architecture before CUDA compiler detection so CMake never injects a generic
# host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  message(FATAL_ERROR
    "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.1)
  message(FATAL_ERROR
    "NInfer requires CUDA 13.1 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()

# --- Host toolchain hardening ------------------------------------------------
# MSVC needs switches that GCC/Clang enable by default. Registering them here keeps
# individual targets clean, and they are mirrored onto the CUDA line because nvcc
# forwards host flags through -Xcompiler rather than inheriting them.
#
# /bigobj is not optional: nvcc emits host translation units that exceed the default
# section limit on the larger operator files.
if(MSVC)
  foreach(_flag /utf-8 /Zc:__cplusplus /bigobj)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${_flag}>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_flag}>)
  endforeach()
endif()

set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(NINFER_BUILD_PROMPT_INPUT OFF)
if(NINFER_BUILD_APPS)
  set(NINFER_BUILD_PROMPT_INPUT ON)
endif()

set(NINFER_BUILD_SERVE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_SERVE ON)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# --- Media dependencies ------------------------------------------------------
# The Linux path locates FFmpeg and libcurl through pkg-config. Windows has no
# pkg-config, so a prebuilt prefix is supplied explicitly and the *same* imported
# target names are synthesised. Every downstream target_include_directories /
# target_link_libraries line therefore stays byte-identical across platforms.
#
# Any distribution that ships include/ and lib/ works: vcpkg, gyan.dev or a BtbN
# FFmpeg build, and the official curl-for-Windows build.
function(ninfer_import_prebuilt_target name)
  cmake_parse_arguments(ARG "" "ROOT" "HEADERS;LIBS" ${ARGN})

  if(NOT ARG_ROOT)
    message(FATAL_ERROR
      "Windows build requires -DNINFER_${name}_ROOT=<prefix> pointing at a prebuilt "
      "${name} tree that contains include/ and lib/.")
  endif()

  foreach(_header IN LISTS ARG_HEADERS)
    if(NOT EXISTS "${ARG_ROOT}/include/${_header}")
      message(FATAL_ERROR
        "NINFER_${name}_ROOT='${ARG_ROOT}' does not contain include/${_header}.")
    endif()
  endforeach()

  set(_libraries "")
  foreach(_lib IN LISTS ARG_LIBS)
    set(_suffixes ".lib" "d.lib" "_a.lib")
    set(_resolved "")
    foreach(_suffix IN LISTS _suffixes)
      if(EXISTS "${ARG_ROOT}/lib/${_lib}${_suffix}")
        set(_resolved "${ARG_ROOT}/lib/${_lib}${_suffix}")
        break()
      endif()
    endforeach()
    if(NOT _resolved)
      message(FATAL_ERROR
        "NINFER_${name}_ROOT='${ARG_ROOT}' has no lib/${_lib}{.lib} under lib/.")
    endif()
    list(APPEND _libraries "${_resolved}")
  endforeach()

  add_library(PkgConfig::${name} INTERFACE IMPORTED)
  set_target_properties(PkgConfig::${name} PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ARG_ROOT}/include"
    INTERFACE_LINK_LIBRARIES "${_libraries}")
  message(STATUS "Using prebuilt ${name} at ${ARG_ROOT}")
endfunction()

if(WIN32)
  ninfer_import_prebuilt_target(FFMPEG
    ROOT "${NINFER_FFMPEG_ROOT}"
    HEADERS "libavformat/avformat.h;libavcodec/avcodec.h;libavutil/avutil.h;libswscale/swscale.h"
    LIBS "avformat;avcodec;avutil;swscale")

  if(NINFER_BUILD_MEDIA_ACQUIRE)
    ninfer_import_prebuilt_target(CURL
      ROOT "${NINFER_CURL_ROOT}"
      HEADERS "curl/curl.h"
      LIBS "libcurl")
  endif()
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
  if(NINFER_BUILD_MEDIA_ACQUIRE)
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
  endif()
endif()

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
