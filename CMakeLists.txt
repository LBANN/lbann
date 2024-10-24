################################################################################
## Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
## LBANN Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
cmake_minimum_required(VERSION 3.27)
project(LBANNv2
  VERSION 0.0.1
  DESCRIPTION "DiHydrogen integration with PyTorch"
  HOMEPAGE_URL "https://github.com/lbann"
  LANGUAGES CXX
)

option(LBANNV2_DEBUG_MODE
  "Enable extra assertions helpful in debugging."
  OFF)

# Make Tom's life easier
set(CMAKE_EXPORT_COMPILE_COMMANDS ON
  CACHE BOOL "Write compile_commands.json" FORCE)

# FIXME (trb): This is probably the right thing, but we should think
# about if this is strictly needed.
set(BUILD_SHARED_LIBS ON)
set(CMAKE_CXX_STANDARD 20) # For DiHydrogen

# FIXME (trb): These are generally useful for development and
# debugging. I should probably pass them on cmd line, but again, lazy.
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -fno-omit-frame-pointer")
set(CMAKE_HIP_FLAGS_DEBUG "-g3 -O0 -fno-omit-frame-pointer")

# Language support
#
# Just set things for CUDA *and* HIP hoping they'll be ignored on
# irrelevant platforms.

# Volta, ampere, hopper
# FIXME (trb): Remove volta ASAP.
set(CMAKE_CUDA_ARCHITECTURES 70 80 90)
set(TORCH_CUDA_ARCH_LIST 7.0 8.0 9.0)
set(CMAKE_CUDA_STANDARD 17)

# MI50, MI250X, MI300A, MI300X
set(CMAKE_HIP_ARCHITECTURES gfx906 gfx90a gfx942)
set(ENV{PYTORCH_ROCM_ARCH} "${CMAKE_HIP_ARCHITECTURES}")
set(PYTORCH_ROCM_ARCH ${CMAKE_HIP_ARCHITECTURES})

# Setup dependencies

set(LBANNV2_MINIMUM_Python_VERSION 3.9)
set(LBANNV2_MINIMUM_H2_VERSION 0.4.0)
set(LBANNV2_MINIMUM_Torch_VERSION 2.6.0)

find_package(Python
  ${LBANNV2_MINIMUM_Python_VERSION}
  REQUIRED
  COMPONENTS Interpreter Development.Module)

# Interrogate the Python environment (via pip) to detect NVIDIA
# dependencies in the environment. Currently, this is based on the
# Torch module that's installed in the environment, if any exists, and
# meaningful values will only be returned if such a module exists.
#
# FIXME (trb): We just handle cuDNN and NCCL here because those are
# the only ones that overlap with Al/H2 needs, but we might consider
# adding paths for the rest of them since Torch will (presumably)
# depend on them.
#
# An alternative approach _could_ be to detect all NVIDIA modules
# known to pip and simply parse those. I'm not sure how realistic this
# might be in practice, but presumably one _could_ have
# nvidia-cudnn-cu11 and nvidia-cudnn-cu12 in the same environment, and
# one could imagine that those packages would provide distinct
# installations of these libraries (fun fact: they don't). Hence the
# preference to let PyTorch tell me which modules it should use. If
# someone was trying to use a Torch that Pip couldn't detect but with
# pip-managed NVIDIA modules, I would classify them as a "power user"
# and expect that they can handle adding command line arguments to the
# LBANNv2 build.
list(PREPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(LBANNv2DetectTorchNVIDIALibraries)
detect_torch_nvidia_libraries(LIBRARIES cudnn nccl)

foreach (pkg cudnn nccl)
  if (LBANNV2_DETECTED_${pkg})
    string(TOUPPER "${pkg}" pkg_upper)
    set(${pkg_upper}_LIBRARY
      "${LBANNV2_DETECTED_${pkg}_LIBRARY}"
      CACHE
      FILEPATH
      "Path to ${pkg_upper} library." FORCE)
    set(${pkg_upper}_INCLUDE_PATH
      "${LBANNV2_DETECTED_${pkg}_INCLUDE_PATH}"
      CACHE
      PATH
      "Include directory for ${pkg}" FORCE)
  endif()
endforeach ()

# Special handling for Torch+cuDNN
if (LBANNV2_DETECTED_cudnn)
  # Torch uses "LIBRARY_PATH" for the location of the main cuDNN
  # library. Because why wouldn't they??
  set(CUDNN_LIBRARY_PATH
    "${LBANNV2_DETECTED_cudnn_LIBRARY}"
    CACHE
    FILEPATH
    "Path to cuDNN library.")

  set(CAFFE2_USE_CUDNN ON CACHE BOOL "Have the build search for cuDNN")
endif ()

# Ok, the CMake here gets a little rocky. The goal is to "pip install
# ." and it should just build "the right thing". So we need to
# auto-detect as much as we can under the weakest assumptions possible
# (e.g., we should not assume "torch.cuda.is_available()" gives
# meaningful information, as we may be building on a GPU-less head
# node). It seems reasonable to just find Torch and see what its CMake
# export can tell us. For instance, "torch_hip" will be found on ROCm
# platforms, and "torch_cuda" will be found on CUDA platforms -- we
# assume (hope!) that these are truly orthogonal! From there, we can
# pull a few additional flags in by further interrogating the targets,
# if needed.

find_package(Torch
  ${LBANNV2_MINIMUM_Torch_VERSION}
  REQUIRED
)

# We also don't care about the limited API nonsense, so we can use
# libtorch. Let's find it.
if (TORCH_LIBRARY)
  get_filename_component(TORCH_LIB_DIR "${TORCH_LIBRARY}" DIRECTORY)
endif ()
find_library(TORCH_PYTHON_LIBRARY
  torch_python
  HINTS
  ${TORCH_LIB_DIR}
  ${Python_SITELIB}/torch/lib64
  ${Python_SITELIB}/torch/lib
  NO_DEFAULT_PATH)
find_library(TORCH_PYTHON_LIBRARY torch_python REQUIRED)

# MI300A only becomes a factor when doing a ROCm build. So start by
# assuming we don't have it.
#
# FIXME (trb): This should, of course, be relaxed to just represent
# memory coherence. However, I don't have access to any non-MI300A
# memory-coherent architectures. If anyone does, I'm happy to abstract
# this now; otherwise, I'll wait until I acquire such access myself.
set(LBANNV2_WITHOUT_MI300A ON)
unset(LBANNV2_WITH_MI300A)
unset(LBANNV2_UNKNOWN_MI300A)

if (TARGET torch_cuda)
  set(ALUMINUM_ENABLE_CUDA ON)
  set(ALUMINUM_ENABLE_NCCL ON)

  set(H2_ENABLE_CUDA ON)

  # We need to edit out the CUDA arch flags out. Or at least edit them
  # down to supported archs (>=70).
elseif (TARGET torch_hip)
  enable_language(HIP)

  set(ALUMINUM_ENABLE_ROCM ON)
  set(ALUMINUM_ENABLE_NCCL ON)

  set(H2_ENABLE_ROCM ON)

  # Handle MI300A configure checks.
  include(LBANNv2DetermineMI300A)
  set(_valid_mi300a_status "WITH" "WITHOUT" "UNKNOWN")
  set(LBANNV2_MI300A_STATUS "DETECT"
    CACHE STRING
    "On MI300A? Valid values: WITH, WITHOUT, UNKNOWN, DETECT")
  string(TOUPPER "${LBANNV2_MI300A_STATUS}" _mi300a_status_upper)
  if (NOT _mi300a_status_upper IN_LIST _valid_mi300a_status)
    determine_mi300a_support(_mi300a_status_upper)
  endif ()
  unset(LBANNV2_WITH_MI300A)
  unset(LBANNV2_WITHOUT_MI300A)
  unset(LBANNV2_UNKNOWN_MI300A)
  set(LBANNV2_${_mi300a_status_upper}_MI300A ON)
  # If we determine that we have MI300A, we can make some static
  # optimizations and eliminate some flow control. In the "UNKNOWN"
  # case, these static branches are replaced by dynamic ones, possibly
  # incurring some small overhead.
  #
  # As far as I can figure, the only case in which this could cause
  # problems (rather than just being suboptimal) is if we declare (or
  # decide) that we have MI300A when we actually do not. In
  # particular, this would cause our assumptions about CPU/GPU memory
  # visibility to be invalid -- hipMalloc'd memory would not be valid
  # on the CPU.

  # We need to remove any "std=c++<XY>" type options because we're
  # ahead of PyTorch's minimum requirements there.
  get_target_property(
    _torch_hip_compile_opts
    torch_hip
    INTERFACE_COMPILE_OPTIONS)
  foreach (_opt ${_torch_hip_compile_opts})
    if (_opt MATCHES "-std=c\\+\\+[0-9a-z]+")
      list(REMOVE_ITEM _torch_hip_compile_opts "${_opt}")
    endif ()
  endforeach()
  set_target_properties(torch_hip
    PROPERTIES INTERFACE_COMPILE_OPTIONS "${_torch_hip_compile_opts}")
endif ()

# We need to determine if we should be using a CXX11_ABI macro or not
# so we can forward as appropriate to spdlog/Catch2/etc. We need to do
# this *BEFORE* adding DiHydrogen(/spdlog/Catch2); otherwise it won't
# get picked up and we'd have to add it to the respective targets
# later on.
if (TORCH_CXX_FLAGS AND TORCH_CXX_FLAGS MATCHES "GLIBCXX_USE_CXX11_ABI=([01])")
  add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${CMAKE_MATCH_1})
endif ()

include(FetchContent)
FetchContent_Declare(
  DiHydrogen
  GIT_REPOSITORY https://github.com/LLNL/DiHydrogen.git
  GIT_TAG 3ecd2a51ad14c257c81cc5e121cf65f9900b7bcf # develop on 28 Jan 2025
  FIND_PACKAGE_ARGS
  NAMES DiHydrogen
  ${LBANNV2_MINIMUM_H2_VERSION}
  COMPONENTS Core Meta Patterns
  CONFIG
)
FetchContent_MakeAvailable(DiHydrogen)

if (DiHydrogen_FOUND)
  message(STATUS "Found DiHydrogen: ${DiHydrogen_DIR}")
  message(STATUS "DiHydrogen version: ${DiHydrogen_VERSION}")
else ()
  message(STATUS "Building DiHydrogen with FetchContent")
endif ()

# Python module stuff
find_package(pybind11 CONFIG REQUIRED)

# Set a few RPATH handling things
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
if(APPLE)
  list(PREPEND CMAKE_INSTALL_RPATH "@loader_path")
else()
  list(PREPEND CMAKE_INSTALL_RPATH "\$ORIGIN")
endif()

# Add the library
add_library(lbannv2 SHARED)
add_library(lbann::lbannv2 ALIAS lbannv2)
target_sources(lbannv2
  PUBLIC
  FILE_SET HEADERS
  BASE_DIRS src
)
target_link_libraries(lbannv2
  PUBLIC
  H2::H2Core
  torch
)
set_target_properties(lbannv2
  PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  VERSION ${LBANNv2_VERSION}
  SOVERSION ${LBANNv2_VERSION_MAJOR}
)

# Create the Python module
python_add_library(_lbannv2 MODULE WITH_SOABI)
target_link_libraries(_lbannv2
  PUBLIC
  lbann::lbannv2
  "${TORCH_PYTHON_LIBRARY}"
  PRIVATE
  pybind11::headers
)
set_target_properties(_lbannv2
  PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)

# Handle logging. If `LBANNV2_LOG_LEVEL` is not set,
# SPDLOG_ACTIVE_LEVEL will not be set on the command line and will
# default to `SPDLOG_LEVEL_TRACE` in the C++ code
# (src/lbannv2/utils/logging.hpp).
#
# NOTE that this is the *compile time* log level. That is, if
# LBANN_LOG_LEVEL is set to "TRACE", every log message (*using the
# LBANNV2_LOG* macros) will be compiled; if it's set to "INFO",
# messages flagged as "TRACE" or "DEBUG" will not even be compiled.
# The default is set to "TRACE" so that all log messages are
# available, depending on the log level selected at runtime.
set(lbannv2_ok_log_levels
  "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF")
if (LBANNV2_LOG_LEVEL IN_LIST lbannv2_ok_log_levels)
  target_compile_definitions(
    lbannv2
    PRIVATE
    SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${LBANNV2_LOG_LEVEL}
  )

  target_compile_definitions(
    _lbannv2
    PRIVATE
    SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${LBANNV2_LOG_LEVEL}
  )
endif ()

# Add the sources to the library
add_subdirectory(src/lbannv2)

# Generate the export header
include(GenerateExportHeader)
generate_export_header(lbannv2)

# Generate the configuration header
configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/lbannv2_config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/lbannv2_config.h
  @ONLY
)

# Include it in the file set
target_sources(lbannv2 PUBLIC
  FILE_SET HEADERS
  BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR}
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/lbannv2_config.h
  ${CMAKE_CURRENT_BINARY_DIR}/lbannv2_export.h
)

# Handle unit testing
include(CTest)
if (BUILD_TESTING)
  add_subdirectory(test)
endif ()

# Install stuff
#
# When building the Python bindings, we still install the whole C++
# library. We might want to clean this up. Also, we set
# tools.scikit-build.wheel.install-dir=lbannv2 so it installs into
# <site-packages>/lbannv2.
include(GNUInstallDirs)

set(
  CMAKE_INSTALL_CMAKEDIR
  "${CMAKE_INSTALL_LIBDIR}/cmake/lbannv2"
)

install(TARGETS lbannv2
  EXPORT lbannv2Targets
  FILE_SET HEADERS
)

install(EXPORT lbannv2Targets
  DESTINATION ${CMAKE_INSTALL_CMAKEDIR}
  NAMESPACE lbann::
)

install(TARGETS _lbannv2
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/lbannv2Config.cmake.in
  "${CMAKE_BINARY_DIR}/lbannv2Config.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_CMAKEDIR}"
)
write_basic_package_version_file(
  lbannv2ConfigVersion.cmake
  COMPATIBILITY SameMinorVersion
)
install(
  FILES
  "${CMAKE_BINARY_DIR}/lbannv2Config.cmake"
  "${CMAKE_BINARY_DIR}/lbannv2ConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_CMAKEDIR}"
)
