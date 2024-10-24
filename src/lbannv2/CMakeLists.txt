################################################################################
## Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
## LBANN Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
target_sources(lbannv2
  PUBLIC
  FILE_SET HEADERS
  FILES
  types.hpp
)

add_subdirectory(backend)
add_subdirectory(memory)
add_subdirectory(ops)
add_subdirectory(utils)

# Pybind/Torch registration
if (SKBUILD)
  add_subdirectory(python)
endif ()
