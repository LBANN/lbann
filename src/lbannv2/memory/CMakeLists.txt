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
  allocator.hpp
  h2_allocator_wrappers.hpp
  registry.hpp
  toplevel_allocator.hpp
)
target_sources(lbannv2
  PRIVATE
  allocator.cpp
  registry.cpp
  toplevel_allocator.cpp
)

if (LBANNV2_UNKNOWN_MI300A OR LBANNV2_WITH_MI300A)
  target_sources(lbannv2
    PUBLIC
    FILE_SET HEADERS
    FILES
    mi300a_allocator.hpp
  )
  target_sources(lbannv2
    PRIVATE
    mi300a_allocator.cpp
  )
endif ()
