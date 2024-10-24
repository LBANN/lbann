////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <cxxabi.h>
#include <execinfo.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace lbannv2
{

inline std::string demngl(std::string symb)
{
  int status;
  char* const demangled_name =
    abi::__cxa_demangle(symb.data(), nullptr, nullptr, &status);
  if (demangled_name && status == 0)
  {
    std::string out(demangled_name);
    free(demangled_name);
    return out;
  }

  std::ostringstream oss;
  oss << symb << " (demangling failed)";
  return oss.str();
}

inline void print_bt(size_t nframes = 128, std::ostream& os = std::cout)
{
  std::vector<void*> frames(nframes);
  nframes = backtrace(frames.data(), nframes);
  char** symbs = backtrace_symbols(frames.data(), nframes);

  os << "-------------------------------------------------\n";
  for (size_t i = 0; i < nframes; ++i)
  {
    os << std::setw(4) << std::right << i << ": (" << frames[i]
       << "): " << demngl(symbs[i]) << "\n";
  }
  os << "-------------------------------------------------" << std::endl;
  free(symbs);
}

}  // namespace lbannv2
