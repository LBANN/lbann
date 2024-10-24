################################################################################
## Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
## LBANN Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
function(detect_torch_nvidia_libraries)
  set(_detect_opts)
  set(_detect_single_val_args)
  set(_detect_multi_value_args LIBRARIES)
  cmake_parse_arguments(PARSE_ARGV 0 _detect
    "${_detect_opts}" "${_detect_single_value_args}" "${_detect_multi_value_args}")

  find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

  # Get information about torch. If Pip doesn't know about torch, that's
  # fine. We just stop and fall back on the user's environment, assuming
  # Torch to have been built from source.
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pip show --no-color torch
    ERROR_VARIABLE _detect_pip_show_error
    OUTPUT_VARIABLE _detect_pip_show_output
    RESULT_VARIABLE _detect_pip_show_result)

  # Split the string on newlines
  string(REPLACE "\n" ";" _detect_torch_show_lines "${_detect_pip_show_output}")

  # And find the "requires" line:
  list(FILTER _detect_torch_show_lines INCLUDE REGEX "^Requires")

  # Now filter that down to the NVIDIA modules:
  string(REGEX MATCHALL "nvidia-[-a-z]+-cu[0-9]+" _detect_nvidia_modules "${_detect_torch_show_lines}")

  # Now that we have a list of modules, we need to search the file lists
  # of these. There are at least 2 approaches.
  #
  #  1. We can interrogate 'pip show --files <module name>', parse the
  #     base from the "Location" line and prepend it to any matching
  #     lines under the "Files" header to get the full paths to any
  #     relevant files.
  #
  #  2. We can use 'importlib.metadata' to parse the metadata associated
  #     with each module. This has the advantage that we don't have to
  #     do as much manual parsing and string manipulation -- the data we
  #     need can be generated with a simple list comprehension.
  #
  # While neither approach is particularly difficult, I've opted for
  # number 2. I especially like that by joining the output string with
  # semicolons, CMake will natively interpret the list of paths as a
  # CMake list, further simplifying things.

  # Get the list of paths out of the metadata. Separate with semicolon
  # so CMake interprets the output as a list directly.
  set(_detect_get_paths_program
    "import importlib.metadata as md; import sys; print(\";\".join([str(f.locate()) for f in md.files(sys.argv[1])]))")

  foreach (lib IN LISTS _detect_LIBRARIES)
    string(REGEX MATCH "nvidia-${lib}-cu[0-9]+" _detect_nvidia_lib_module "${_detect_nvidia_modules}")

    # Find paths
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -c "${_detect_get_paths_program}" "${_detect_nvidia_lib_module}"
      ERROR_VARIABLE _detect_get_paths_error
      OUTPUT_VARIABLE _detect_get_paths_output
      RESULT_VARIABLE _detect_get_paths_result)

    foreach (path ${_detect_get_paths_output})
      if (path MATCHES ".*${lib}\\.h$")

        cmake_path(GET
          path
          PARENT_PATH
          _detect_parent_path)
        set(LBANNV2_DETECTED_${lib}_INCLUDE_PATH
          "${_detect_parent_path}"
          CACHE
          PATH
          "Include directory for ${lib}")

      elseif (path MATCHES ".*lib${lib}${CMAKE_SHARED_LIBRARY_SUFFIX}.*")

        set(LBANNV2_DETECTED_${lib}_LIBRARY
          "${path}"
          CACHE
          FILEPATH
          "Library for ${lib}")

      endif ()
    endforeach ()

    # Consider the thing found if both the include path and the
    # library are available.
    if (LBANNV2_DETECTED_${lib}_LIBRARY AND LBANNV2_DETECTED_${lib}_INCLUDE_PATH)
      set(LBANNV2_DETECTED_${lib} TRUE PARENT_SCOPE)
    endif ()
  endforeach ()
endfunction ()
