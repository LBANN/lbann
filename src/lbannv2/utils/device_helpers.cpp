////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include "lbannv2_config.h"

#include "device_helpers.hpp"

#include "lbannv2/backend/library_state.hpp"
#include "lbannv2/utils/errors.hpp"

#if LBANNV2_HAS_GPU
#include <h2/gpu/runtime.hpp>
#endif

// FIXME (trb): At this time, these are implemented VERY permissively.
// "to_native" accepts a device without an index and, in such a case,
// it will query the library state and return the currently selected
// device (in its native c10/Torch representation). Similarly,
// "to_lbann" accepts GPU device types without index. In this case, it
// will simply return the LBANN GPU device. It will, however, throw if
// it's given an indexed GPU type where the index does not match
// DiHydrogen's currently selected GPU.

c10::Device lbannv2::to_native(c10::Device const& lbann_device)
{
  if (!is_lbann(lbann_device))
    return lbann_device;

  auto const idx = lbann_device.index();
  if (idx < 0)  // "Use the current device"
    return state::current_device_native();

  LBANNV2_ASSERT(
    idx < NumLBANNDevices,
    std::runtime_error,
    "Invalid device index. At this time, LBANNv2 only supports CPU and "
    "a single GPU (per MPI rank). device=\"lbann:0\" denotes the CPU "
    "and device=\"lbann:1\" denotes the GPU (CUDA or ROCm platforms "
    "only). LBANNv2 will use the device returned by state::gpu_idx().");

  if (idx == LBANN_CPU)
    return c10::Device {c10::DeviceType::CPU};

#if LBANNV2_HAS_GPU
  // if (idx == LBANN_GPU)
  return {LBANN_GPU_TYPE, state::gpu_idx()};
#else
  throw std::runtime_error("Invalid device index");
#endif
}

c10::Device lbannv2::to_lbann(c10::Device const& c10_device)
{
  if (is_lbann(c10_device))
    return c10_device;

  switch (c10_device.type())
  {
  case c10::DeviceType::CPU: return {LBANNDeviceT, LBANN_CPU};
#if LBANNV2_HAS_GPU
  case c10::DeviceType::CUDA:
#if LBANNV2_HAS_ROCM
  case c10::DeviceType::HIP:
#endif
  {
    LBANNV2_ASSERT(
      c10_device.index() < 0  // "current GPU"
        || c10_device.index() == state::gpu_idx(),
      std::runtime_error,
      "Invalid GPU index. If provided, the GPU index must match the index "
      "returned by state::gpu_idx().");
    return {LBANNDeviceT, LBANN_GPU};
  }
#endif
  default: throw std::runtime_error("Device type not handled by LBANN");
  }
}

c10::DispatchKeySet lbannv2::get_default_keyset(c10::Device const& d)
{
  switch (d.type())
  {
  case c10::kCPU: return c10::DispatchKeySet{c10::DispatchKey::CPU};
  case c10::kCUDA: return c10::DispatchKeySet{c10::DispatchKey::CUDA};
  case c10::kPrivateUse1: return c10::DispatchKeySet{c10::DispatchKey::PrivateUse1};
  default:
    throw std::runtime_error("Unknown device type");
  }
}
