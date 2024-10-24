////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "lbannv2_config.h"

#include <catch2/catch_test_macros.hpp>

#if LBANNV2_WITH_MI300A
#define SKIP_WHEN_NO_MI300A()
#elif LBANNV2_WITHOUT_MI300A
#define SKIP_WHEN_NO_MI300A() SKIP("No MI300A support")
#elif LBANNV2_UNKNOWN_MI300A
#include <h2/gpu/runtime.hpp>
#define SKIP_WHEN_NO_MI300A()                                                  \
  do                                                                           \
  {                                                                            \
    if (!h2::gpu::is_integrated())                                             \
    {                                                                          \
      SKIP("No MI300A support");                                               \
    }                                                                          \
  } while (0)
#endif
