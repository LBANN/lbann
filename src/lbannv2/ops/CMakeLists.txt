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
  copy.hpp
  empty_tensor.hpp
)
target_sources(lbannv2
  PRIVATE
  copy.cpp
  empty_tensor.cpp
)

# Note that LBANNV2_HAS_ROCM is implicit in either of these cases.
#
# FIXME trb: "migrate" includes all the dynamic mi300a handling, etc.
# Should it always be available at this level? (vs just in
# register_ops.cpp)
if (LBANNV2_UNKNOWN_MI300A OR LBANNV2_WITH_MI300A)
  target_sources(lbannv2
    PUBLIC
    FILE_SET HEADERS
    FILES
    migrate.hpp
    nonzero.hpp
    scalar.hpp
  )
  target_sources(lbannv2
    PRIVATE
    migrate.cpp
    nonzero.hip
    scalar.cpp
  )
endif ()
