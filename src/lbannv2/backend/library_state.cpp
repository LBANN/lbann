////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include "lbannv2/backend/library_state.hpp"

namespace lbannv2
{
LibState& LibState::instance()
{
#if LBANNV2_HAS_GPU
  static LibState state {static_cast<c10::DeviceIndex>(h2::gpu::current_gpu())};
#else
  static LibState state;
#endif
  return state;
}
}  // namespace lbannv2
