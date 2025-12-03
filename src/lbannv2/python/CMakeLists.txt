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
)

if (LBANNV2_WITH_MI300A OR LBANNV2_UNKNOWN_MI300A)
  target_sources(_lbannv2
    PRIVATE
    register_mi300a_ops.cpp
  )
endif ()
