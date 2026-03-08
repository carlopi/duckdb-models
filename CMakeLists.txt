cmake_minimum_required(VERSION 3.10)

# Set extension name here
set(TARGET_NAME models)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})
include_directories(src/include)

set(EXTENSION_SOURCES src/models_extension.cpp)

# llama.cpp backend — required (via vcpkg toolchain).
# vcpkg places the llama config under share/llama-cpp/ (not share/llama/),
# so we need explicit HINTS. We look in:
#   1. <project_root>/vcpkg/installed/<triplet>/share/llama-cpp  (classic mode)
#   2. <VCPKG_INSTALLED_DIR>/<triplet>/share/llama-cpp           (manifest mode)
set(_vcpkg_classic "${CMAKE_SOURCE_DIR}/vcpkg/installed")
set(_triplet "${VCPKG_TARGET_TRIPLET}")
if(NOT _triplet)
  if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm|ARM")
    set(_triplet "arm64-osx")
  elseif(APPLE)
    set(_triplet "x64-osx")
  else()
    set(_triplet "x64-linux")
  endif()
endif()
set(_llama_hints
    "${_vcpkg_classic}/${_triplet}/share/llama-cpp"
    "${VCPKG_INSTALLED_DIR}/${_triplet}/share/llama-cpp")

# Make sure the classic-mode vcpkg prefix is on CMAKE_PREFIX_PATH so that
# llama's transitive find_package(ggml) also works.
list(APPEND CMAKE_PREFIX_PATH "${_vcpkg_classic}/${_triplet}")

# llama-config.cmake uses set_and_check() on a bin/ dir that static-only
# vcpkg triplets (e.g. arm64-linux-release) never create. Pre-create it so
# the config file doesn't abort. VCPKG_INSTALLED_DIR / VCPKG_TARGET_TRIPLET
# are injected by the vcpkg cmake toolchain and give us the exact path.
if(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
  file(MAKE_DIRECTORY "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin")
endif()

find_package(llama CONFIG HINTS ${_llama_hints})
if(llama_FOUND)
  message(STATUS "llama.cpp found")
  list(APPEND EXTENSION_SOURCES src/llama_backend.cpp)
else()
  message(FATAL_ERROR "llama.cpp not found — required by the models extension")
endif()

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
set(PARAMETERS "-no-warnings")
build_loadable_extension(${TARGET_NAME} ${PARAMETERS} ${EXTENSION_SOURCES})

target_link_libraries(${EXTENSION_NAME}          llama)
target_link_libraries(${LOADABLE_EXTENSION_NAME} llama)
if(APPLE)
  target_link_libraries(${EXTENSION_NAME}          "-framework Metal" "-framework Accelerate")
  target_link_libraries(${LOADABLE_EXTENSION_NAME} "-framework Metal" "-framework Accelerate")
endif()

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
