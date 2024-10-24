////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

// FIXME (trb): Where should this file live??

#include <c10/core/ScalarType.h>

namespace lbannv2
{

/** @brief Decide if a data type is supported by LBANNv2. */
inline bool is_supported(c10::ScalarType t) noexcept
{
  switch (t)
  {
  case c10::ScalarType::Bool:
  case c10::ScalarType::Float:
  case c10::ScalarType::Double:
  case c10::ScalarType::Int:
  case c10::ScalarType::UInt32:
  case c10::ScalarType::Long: return true;
  default: return false;
  }
}

}  // namespace lbannv2
