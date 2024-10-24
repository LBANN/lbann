################################################################################
## Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
## LBANN Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
if (NOT SKBUILD)
  message(FATAL_ERROR "You should not be here. Not doing a SKBUILD.")
endif ()

target_sources(_lbannv2
  PRIVATE
  register_lbannv2.cpp
  register_memory_funcs.cpp
  register_ops.cpp
  register_torch_support.cpp
)
