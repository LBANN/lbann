////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <lbannv2_config.h>

#include <h2/core/device.hpp>

#include <c10/core/Device.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>

namespace lbannv2
{

inline constexpr c10::DeviceIndex LBANN_CPU = 0;
inline constexpr c10::DeviceType LBANN_CPU_TYPE = c10::DeviceType::CPU;
#if LBANNV2_HAS_GPU
inline constexpr c10::DeviceIndex LBANN_GPU = 1;
inline constexpr c10::DeviceIndex NumLBANNDevices = 2;

// NOTE: Big errors occur if this is c10::DeviceType::HIP. Always CUDA
// all the time always.
inline constexpr c10::DeviceType LBANN_GPU_TYPE = c10::DeviceType::CUDA;
#else
inline constexpr c10::DeviceIndex NumLBANNDevices = 1;
#endif

inline constexpr c10::DeviceType LBANNDeviceT = c10::DeviceType::PrivateUse1;
inline constexpr c10::DispatchKey LBANNDispKey = c10::DispatchKey::PrivateUse1;
inline constexpr c10::BackendComponent LBANNBit =
  c10::BackendComponent::PrivateUse1Bit;

inline bool is_lbann(c10::Device const& d) noexcept
{
  return d.is_privateuseone();
}

/** @brief Convert an LBANN c10::Device to one with a native
 *         c10::DeviceType.
 *
 *  The input device must have a valid index.
 */
c10::Device to_native(c10::Device const& device);

/** @brief Convert a native c10::Device to the corresponding LBANN
 *         c10::Device.
 *
 *  The returned device will have `type() == LBANNDeviceT` and an
 *  index of 0 (CPU) or 1 (GPU, if enabled). LBANN does not
 *  distinguish HIP from CUDA. Other input device types will throw an
 *  exception.
 */
c10::Device to_lbann(c10::Device const& device);

/** @brief Get the LBANN c10::DeviceIndex for the given c10::Device.
 *
 *  If the input device is an LBANN c10::Device, its index will be
 *  returned unmodified. Otherwise, CPU is index 0 and any GPU is
 *  index 1. Other device types will throw an exception.
 */
c10::DeviceIndex to_lbann_index(c10::Device const& device);

/** @brief Get the dispatch keyset for the given device. */
c10::DispatchKeySet get_default_keyset(c10::Device const& c10_device);

}  // namespace lbannv2
