////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <lbannv2_config.h>

#define LBANNV2_ASSERT(cond, excpt, msg)                                       \
  do                                                                           \
  {                                                                            \
    if (!(cond))                                                               \
    {                                                                          \
      throw excpt(msg);                                                        \
    }                                                                          \
  } while (0)

#define LBANNV2_ASSERT_ALWAYS(cond)                                            \
  LBANNV2_ASSERT(cond, std::runtime_error, "Assertion \"" #cond "\" failed.")

#if LBANNV2_DEBUG
#define LBANNV2_ASSERT_DEBUG(cond) (void)
#else
#define LBANNV2_ASSERT_DEBUG(cond) LBANNV2_ASSERT_ALWAYS(cond)
#endif
