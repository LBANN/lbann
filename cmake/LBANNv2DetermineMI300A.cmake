################################################################################
## Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
## LBANN Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
cmake_minimum_required(VERSION 3.24.0)

# Tries to determine whether the machine in question is MI300A. The
# check hinges on "rocm-smi" returning sane things. Sadly we must do
# this rather than just testing the arch flag because "gfx942" refers
# to both the MI300A and the MI300X.
#
# The return value is ternary:
#
#   - "WITH" means we have determined that we have MI300A
#   - "WITHOUT" means we have determined that we do NOT have MI300A
#   - "UNKNOWN" means that we cannot determine whether we have MI300A,
#     generally because "rocm-smi" produced no usable output.
#
# In cases where the node on which LBANNv2 is being built does not
# have GPUs, or if it happens to have *different* GPUs from the ones
# on the compute nodes, users are advised to provide this information
# directly, if possible.
#
# As this all comes down to "rocm-smi" output on the node on which
# LBANNv2 is configured, users are advised that there is high risk for
# incorrect or suboptimal information if not configuring on a compute
# node.
function (determine_mi300a_support OUTPUT_VARIABLE)

  # Call rocm-smi (should be in the PATH). If rsmi fails, then we say "unknown".
  execute_process(
    COMMAND rocm-smi --showproductname --json
    OUTPUT_VARIABLE _rsmi_info
    ERROR_VARIABLE _rsmi_error
    ERROR_QUIET
  )

  if (_rsmi_error AND _rsmi_error MATCHES ".*ERROR.*")
    set(${OUTPUT_VARIABLE} "UNKNOWN" PARENT_SCOPE)
    return ()
  endif ()

  string(JSON _gfx_version
    ERROR_VARIABLE _json_err
    GET "${_rsmi_info}" "card0" "GFX Version")

  # To get here, rsmi returned something valid, and this path just was not right.
  if (_json_err)
    message(DEBUG
      "JSON Error: ${_json_err}\n\nAssuming MI300A status is 'UNKNOWN'.")
    set(${OUTPUT_VARIABLE} "UNKNOWN" PARENT_SCOPE)
    return ()
  endif ()

  if (_gfx_version MATCHES ".*gfx942.*")
    execute_process(
      COMMAND rocminfo
      OUTPUT_VARIABLE _rocminfo_output
      ERROR_VARIABLE _rocminfo_error
      ERROR_QUIET
    )
    string(FIND "${_rocminfo_output}" "MI300A" _mi300a_exists)
    if (_mi300a_exists EQUAL -1)
      set(${OUTPUT_VARIABLE} "WITHOUT" PARENT_SCOPE)
    else ()
      set(${OUTPUT_VARIABLE} "WITH" PARENT_SCOPE)
    endif ()
  else ()
    set(${OUTPUT_VARIABLE} "WITHOUT" PARENT_SCOPE)
  endif ()
endfunction ()
