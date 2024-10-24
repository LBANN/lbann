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
  device_guard.hpp
  fallback.hpp
  hooks_interface.hpp
  library_state.hpp
)
target_sources(lbannv2
  PRIVATE
  fallback.cpp
  hooks_interface.cpp
  library_state.cpp
)
