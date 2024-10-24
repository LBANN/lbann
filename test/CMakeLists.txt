################################################################################
## Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
## LBANN Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2
  GIT_TAG fa43b77429ba76c462b1898d6cd2f2d7a9416b14 # v3.7.1
  FIND_PACKAGE_ARGS 3.0.0 CONFIG)
FetchContent_MakeAvailable(Catch2)

add_executable(catch-tests
  cpp/test_device_helpers.cpp
  cpp/test_empty_tensor.cpp
  cpp/test_pointer_registry.cpp
  cpp/test_tensor_helpers.cpp
)

if (LBANNV2_UNKNOWN_MI300A OR LBANNV2_WITH_MI300A)
  target_sources(catch-tests
    PRIVATE
    cpp/test_mi300a_allocator.cpp
  )
endif ()

target_link_libraries(catch-tests
  PRIVATE
  lbann::lbannv2
  Catch2::Catch2WithMain
)

set_target_properties(catch-tests
  PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS ON
)
