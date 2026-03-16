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
  debugging_helpers.hpp
  errors.hpp
  gpu_utils.hpp
  logging.hpp
  tensor_helpers.hpp
)
target_sources(lbannv2
  PRIVATE
  gpu_utils.cpp
  logging.cpp
)
