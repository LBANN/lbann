////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <lbannv2_config.h>

#include <lbannv2/types.hpp>
#include <lbannv2/utils/device_helpers.hpp>
#include <lbannv2/utils/errors.hpp>

#if LBANNV2_HAS_GPU
#include <h2/gpu/runtime.hpp>
#endif

#include <atomic>

#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>

// Note that H2 does not presently have a runtime notion of
// "datatype". We should consider how we want to express that.

// Note that we ought not collapse this code into, say, the
// DeviceGuard directly. It would all have to be mutable state but
// this exists more like a CUDA context or something.

namespace lbannv2
{

/** @class LibState
 *  @brief A class for tracking settings related to LBANN.
 */
class LibState
{
  std::atomic<c10::ScalarType> m_current_dtype = c10::ScalarType::Float;
  std::atomic<c10::DeviceType> m_current_device = LBANN_CPU_TYPE;
  c10::DeviceIndex m_gpu_idx = -1;

  /** @name Singleton lifecycle */
  ///@{

  LibState(c10::DeviceIndex gpu_idx = -1) : m_gpu_idx {gpu_idx} {}
  LibState(LibState const&) = delete;
  LibState& operator=(LibState const&) = delete;
  LibState(LibState&&) = delete;
  LibState& operator=(LibState&&) = delete;

  ///@}

public:
  /** @brief Get the library state instance.
   *
   *  This class is thread-safe; obviously use references carefully.
   */
  static LibState& instance();

  static constexpr bool has_gpu() noexcept { return LBANNV2_HAS_GPU; }

  c10::ScalarType current_dtype() const noexcept
  {
    return m_current_dtype.load();
  }
  c10::DeviceType current_device_type() const noexcept
  {
    return m_current_device.load();
  }

  c10::Device current_device_lbann() const noexcept
  {
    return c10::Device {LBANNDeviceT, get_device_idx()};
  }

  c10::Device current_device_native() const noexcept
  {
    auto const device = m_current_device.load();
#if LBANNV2_HAS_GPU
    if (device == c10::DeviceType::CUDA || device == c10::DeviceType::HIP)
      return c10::Device {device, get_gpu_idx()};
#endif
    return c10::Device {device};
  }

  // The class invariants here are such that the only possible values
  // for m_current_device are CPU, HIP, or CUDA. An exception will be
  // thrown in set_device() if any other value is attempted.
  c10::DeviceIndex get_device_idx() const noexcept
  {
#if LBANNV2_HAS_GPU
    return m_current_device.load() != LBANN_CPU_TYPE;
#else
    return 0;
#endif  // LBANNV2_HAS_GPU
  }

  /** @brief Set the current device. */
  void set_device(c10::Device const& d);

  /** @brief Set the current default datatype. */
  void set_type(c10::ScalarType const& t)
  {
    LBANNV2_ASSERT_ALWAYS(is_supported(t));
    m_current_dtype = t;
  }

  c10::DeviceIndex get_gpu_idx() const noexcept { return m_gpu_idx; }
};  // class LBANNv2LibState

namespace state
{

inline c10::ScalarType current_dtype() noexcept
{
  return LibState::instance().current_dtype();
}

inline c10::DeviceType current_device_type() noexcept
{
  return LibState::instance().current_device_type();
}

inline c10::Device current_device_lbann() noexcept
{
  return LibState::instance().current_device_lbann();
}

inline c10::Device current_device_native() noexcept
{
  return LibState::instance().current_device_native();
}

inline c10::DeviceIndex current_device_idx() noexcept
{
  return LibState::instance().get_device_idx();
}

inline c10::DeviceIndex gpu_idx() noexcept
{
  return LibState::instance().get_gpu_idx();
}

inline constexpr bool has_gpu() noexcept
{
  return LibState::has_gpu();
}

inline void set_device(c10::Device const& d)
{
  LibState::instance().set_device(d);
}

inline void set_type(c10::ScalarType const& t)
{
  LibState::instance().set_type(t);
}

}  // namespace state

// This impl is longer due to the (relatively) large amount of preprocessor and
// assertion code. So I'm moving it out-of-line.
inline void LibState::set_device(c10::Device const& d)
{
  LBANNV2_ASSERT(
    is_lbann(d), std::runtime_error, "Device should be LBANN (PrivateUse1).");

  auto const idx = d.index();
  if (idx < 0)
    return;

  LBANNV2_ASSERT(0 <= idx && idx < NumLBANNDevices,
                 std::runtime_error,
                 "Device must have a valid index.");

  if (idx == 0)
    m_current_device.store(LBANN_CPU_TYPE);
#if LBANNV2_HAS_GPU
  else if (idx == 1)
    m_current_device.store(LBANN_GPU_TYPE);
#endif  // LBANNV2_HAS_GPU
}

}  // namespace lbannv2
