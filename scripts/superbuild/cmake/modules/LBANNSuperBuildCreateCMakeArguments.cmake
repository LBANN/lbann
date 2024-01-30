################################################################################
## Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
## Produced at the Lawrence Livermore National Laboratory.
## Written by the LBANN Research Team (B. Van Essen, et al.) listed in
## the CONTRIBUTORS file. <lbann-dev@llnl.gov>
##
## LLNL-CODE-697807.
## All rights reserved.
##
## This file is part of LBANN: Livermore Big Artificial Neural Network
## Toolkit. For details, see http://software.llnl.gov/LBANN or
## https://github.com/LLNL/LBANN.
##
## Licensed under the Apache License, Version 2.0 (the "Licensee"); you
## may not use this file except in compliance with the License.  You may
## obtain a copy of the License at:
##
## http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
## implied. See the License for the specific language governing
## permissions and limitations under the license.
################################################################################
# This function takes a list of variables and writes a CMake argument string of the format:
#   -DARGNAME:ARGTYPE=ARGVALUE ...
#
# Arguments:
#   -- REMOVE_PKG_NAME: <PACKAGE_NAME>_<OPTION> gets written as -D<OPTION>=<...>
#
#   -- PACKAGE_NAME: The name of the package
#   -- OUTPUT_VARIABLE: The list of CMake options
#
#   -- VARIABLES: The list of variable names to process
#   -- EXTRA_REMOVE_PREFIXES: Also strip these prefixes. Takes
#        precendence over SKIP_VARS_WITH_PREFIXES and PACKAGE_NAME.
#   -- SKIP_VARS_WITH_PREFIXES: Skip a variable with this prefix
#        unless caught by EXTRA_REMOVE_PREFIXES
#
function (create_cmake_arguments)
  set(_OPTIONS REMOVE_PKG_NAME)
  set(_ONE_VALUE_PARAMS PACKAGE_NAME OUTPUT_VARIABLE)
  set(_MULTI_VALUE_PARAMS
    VARIABLES SKIP_VARS_WITH_PREFIXES EXTRA_REMOVE_PREFIXES)

  cmake_parse_arguments(_CREATEARGS
    "${_OPTIONS}" "${_ONE_VALUE_PARAMS}" "${_MULTI_VALUE_PARAMS}" ${ARGN})

  # Short-circuit. IDK if is the best decision...
  if (${_CREATEARGS_OUTPUT_VARIABLE})
    return()
  endif ()

  foreach (_variable ${_CREATEARGS_VARIABLES})

    # Check the variable's type, if possible.
    get_property(_CMAKE_ARG_TYPE CACHE ${_variable} PROPERTY TYPE)

    if (_CMAKE_ARG_TYPE STREQUAL "STATIC"
        OR _CMAKE_ARG_TYPE STREQUAL "INTERNAL")
      continue()
    endif ()

    set(_variable_done FALSE)

    # EXTRA_REMOVE_PREFIXES takes precedence over everything
    foreach (_prefix ${_CREATEARGS_EXTRA_REMOVE_PREFIXES})
      string(REGEX REPLACE "^${_prefix}_\(.+\)" "\\1"
        _CMAKE_ARG_NAME ${_variable})
      if (CMAKE_MATCH_COUNT GREATER 0)
        set(_variable_done TRUE)
        break ()
      endif ()
    endforeach ()

    # Check for skips
    if (NOT _variable_done AND _CREATEARGS_SKIP_VARS_WITH_PREFIXES)

      foreach (_prefix ${_CREATEARGS_SKIP_VARS_WITH_PREFIXES})
        string(REGEX MATCH "^${_prefix}_" _MATCH_FOUND ${_variable})
        if (_MATCH_FOUND)
          set(_variable_done TRUE)
          break ()
        endif ()
      endforeach ()

      if(_variable_done)
        continue ()
      endif ()
    endif ()

    if (NOT _variable_done)
      # Cleanup the variable name
      if (_CREATEARGS_REMOVE_PKG_NAME)
        # We must be careful to only remove the first instance of the
        # package name.
        string(REGEX REPLACE "^${_CREATEARGS_PACKAGE_NAME}_\(.+\)" "\\1"
          _CMAKE_ARG_NAME ${_variable})
      else ()
        # Handle CMake options
        string(REGEX REPLACE "${_CREATEARGS_PACKAGE_NAME}_\(CMAKE_.+\)" "\\1"
          _CMAKE_ARG_NAME ${_variable})
      endif ()
      set(_variable_done TRUE)
    endif ()

    # Add the variable to the CMake line
    if (${_variable} MATCHES ".*\;.*") # If it's a list
      string(REPLACE ";" "|" ${_variable} "${${_variable}}")
    endif ()

    if (NOT ${_CMAKE_ARG_TYPE} STREQUAL "UNINITIALIZED")
      list(APPEND _output_string
        "-D${_CMAKE_ARG_NAME}:${_CMAKE_ARG_TYPE}=${${_variable}}")
    else ()
      list(APPEND _output_string "-D${_CMAKE_ARG_NAME}=${${_variable}}")
    endif ()
  endforeach ()

  # Return
  set(${_CREATEARGS_OUTPUT_VARIABLE} ${_output_string} PARENT_SCOPE)

endfunction(create_cmake_arguments)
