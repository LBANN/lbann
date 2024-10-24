////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "lbannv2/utils/device_helpers.hpp"
#include <lbannv2/backend/library_state.hpp>

#include <h2/core/device.hpp>
#include <h2/utils/Error.hpp>

#include <stdexcept>

#include <c10/core/DeviceGuard.h>

namespace lbannv2
{

/** @class DeviceGuardImpl
 *  @brief DeviceGuardImplInterface impl for LBANN
 *
 *  The LBANN device is weird. The device index is meaningful: 0 is
 *  the CPU device, and 1 is the GPU whose CUDA/HIP index is the one
 *  returned by a call to state::gpu_idx() any time after the state
 *  has been initialized. The reason we do not encode a specific GPU
 *  index in the LBANN device tags is that we expect deviceCount() to
 *  be 1 (CPU-only) or 2 (CPU+GPU).
 */
class DeviceGuardImpl final : public c10::impl::DeviceGuardImplInterface
{
public:
  DeviceGuardImpl() = default;
  DeviceGuardImpl(c10::Device d) { setDevice(std::move(d)); }

  c10::DeviceType type() const final { return LBANNDeviceT; }

  c10::Device exchangeDevice(c10::Device d) const final
  {
    c10::Device const old = getDevice();
    setDevice(std::move(d));
    return old;
  }

  c10::Device getDevice() const final { return state::current_device_lbann(); }

  void setDevice(c10::Device d) const final { state::set_device(d); }

  void uncheckedSetDevice(c10::Device d) const noexcept final
  {
    try
    {
      setDevice(d);
    }
    catch (std::runtime_error const&)  // Just ignore the error, if any.
    {}
  }

  c10::Stream getStream(c10::Device d) const noexcept final
  {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }

  c10::Stream getNewStream(c10::Device d, int /*priority*/ = 0) const final
  {
    LBANNV2_ASSERT(
      is_lbann(d), std::runtime_error, "Device must be LBANN (PrivateUse1)");
    return c10::Stream(c10::Stream::DEFAULT, getDevice());
  }

  c10::Stream exchangeStream(c10::Stream) const noexcept final
  {
    return c10::Stream(c10::Stream::DEFAULT, getDevice());
  }

  c10::DeviceIndex deviceCount() const noexcept final
  {
    return state::has_gpu() ? 2 : 1;
  }

  void record(void**,
              c10::Stream const&,
              c10::DeviceIndex const,
              c10::EventFlag const) const final
  {
    throw std::runtime_error("LBANN backend doesn't support events (yet)");
  }

  void block(void*, c10::Stream const&) const final
  {
    throw std::runtime_error("LBANN backend doesn't support events (yet)");
  }

  bool queryEvent(void*) const final
  {
    throw std::runtime_error("LBANN backend doesn't support events (yet)");
  }

  void destroyEvent(void*, c10::DeviceIndex const) const noexcept final
  {
    // nothing
  }

  bool queryStream(c10::Stream const&) const final { return true; }

  void synchronizeStream(c10::Stream const&) const final
  {
    // nothing
  }

private:
};  // struct LBANNv2DeviceGuard

// Quick-and-dirty impl. Per Torch docs, we might want to make this a
// full-fledged wrapper class.
using LBANNDeviceGuard = c10::impl::InlineDeviceGuard<DeviceGuardImpl>;

}  // namespace lbannv2
