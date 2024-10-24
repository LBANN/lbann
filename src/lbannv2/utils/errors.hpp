////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <h2/utils/Error.hpp>

#define LBANNV2_ASSERT(...) H2_ASSERT(__VA_ARGS__)

#define LBANNV2_ASSERT_ALWAYS(cond)                                            \
  H2_ASSERT_ALWAYS(cond, "Assertion \"" #cond "\" failed.")

#define LBANNV2_ASSERT_DEBUG(cond)                                             \
  H2_ASSERT_DEBUG(cond, "Assertion \"" #cond "\" failed.")
