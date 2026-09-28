add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
target_include_directories(ninfer PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_load_progress
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
target_include_directories(ninfer-serve PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_load_progress)

# --- Runtime dependencies beside the executables ---------------------------------------
# Both apps link the FFmpeg and libcurl import libraries, so the matching DLLs must sit
# next to the executable or be on PATH when it starts. Without this the process dies
# before main() with 0xC0000135 (STATUS_DLL_NOT_FOUND), reporting no output at all:
#   avformat-63.dll, avcodec-63.dll, avutil-61.dll, swscale-10.dll, libcurl.dll
# The imported-target helper points the linker at <prefix>/lib, so the DLLs are in
# <prefix>/bin. NINFER_FFMPEG_ROOT and NINFER_CURL_ROOT normally name the same vcpkg
# prefix, so deduplicate before copying.
if(WIN32)
  set(_ninfer_runtime_prefixes "")
  foreach(_r "${NINFER_FFMPEG_ROOT}" "${NINFER_CURL_ROOT}")
    if(_r AND NOT _r IN_LIST _ninfer_runtime_prefixes)
      list(APPEND _ninfer_runtime_prefixes "${_r}")
    endif()
  endforeach()

  foreach(_app ninfer ninfer-serve)
    foreach(_prefix IN LISTS _ninfer_runtime_prefixes)
      if(EXISTS "${_prefix}/bin")
        add_custom_command(TARGET ${_app} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_directory
                  "${_prefix}/bin" "$<TARGET_FILE_DIR:${_app}>"
          COMMENT "Copying runtime DLLs from ${_prefix}/bin next to ${_app}")
      endif()
    endforeach()
  endforeach()
endif()

# --- Embeddable C ABI (host processes such as ComfyUI's Python runtime) ------------------
# A plain DLL/so with extern "C" exports. ctypes / cffi load it into ANY Python process, so
# it is independent of the host's Python version (unlike a CPython extension module). The
# DLL is loaded *into* the host process and shares its CUDA context.
#
# Notes for the first build:
#   * ninfer_engine and its dependencies are STATIC. Linking them into a SHARED target is
#     what produces a self-contained DLL; CMake carries the private static dependencies
#     onto the link line automatically.
#   * The engine's CUDA archives use separable compilation, so device symbols must be
#     resolved when the shared object is linked. The properties below request that. This
#     is the part most likely to need adjustment against a specific CUDA/MSVC combination.
#   * Media (FFmpeg, libcurl) is reached transitively through
#     ninfer_engine -> ninfer_media_decode, so the av*/libcurl import libraries still
#     appear on this link line. To ship a text-only DLL without them, make that dependency
#     conditional in src/CMakeLists.txt.
add_library(ninfer_capi SHARED
  capi/ninfer_capi.cpp)
target_include_directories(ninfer_capi PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(ninfer_capi PRIVATE ninfer_engine)
set_target_properties(ninfer_capi PROPERTIES
  PREFIX ""
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# A ctypes host loads ninfer_capi.dll directly, so its own imports (avformat-*.dll,
# avcodec-*.dll, avutil-*.dll, swscale-*.dll, libcurl.dll) must be resolvable too.
# Without this they are simply absent from the DLL's directory and the load fails with
# a bare "DLL load failed" — reuse the same runtime-prefix list the executables use.
if(WIN32)
  foreach(_prefix IN LISTS _ninfer_runtime_prefixes)
    if(EXISTS "${_prefix}/bin")
      add_custom_command(TARGET ninfer_capi POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${_prefix}/bin" "$<TARGET_FILE_DIR:ninfer_capi>"
        COMMENT "Copying runtime DLLs next to ninfer_capi")
    endif()
  endforeach()
endif()
