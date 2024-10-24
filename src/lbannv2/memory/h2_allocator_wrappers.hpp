////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <lbannv2_config.h>

#include <lbannv2/memory/allocator.hpp>
#include <lbannv2/utils/device_helpers.hpp>
#include <lbannv2/utils/logging.hpp>

#include <h2/core/allocator.hpp>

#include <c10/core/Allocator.h>

namespace lbannv2
{

template <h2::Device D>
class H2AllocatorWrapper : public Allocator
{
  using AllocatorType = h2::internal::Allocator<std::byte, D>;

public:
  /** @name Virtual function overrides */
  ///@{

  // memcpy
  void copy_data(void* dst, void const* src, size_t n) const final
  {
    if constexpr (D == h2::Device::CPU)
    {
      LBANNV2_TRACE("H2AllocatorWrapper<CPU>::copy_data(dst={}, src={}, bytes={})",
                    dst, src, n);
      std::memcpy(dst, src, n);
    }
#if LBANNV2_HAS_GPU
    if constexpr (D == h2::Device::GPU)
    {
      LBANNV2_TRACE("H2AllocatorWrapper<GPU>::copy_data(dst={}, src={}, bytes={})",
                    dst, src, n);
      h2::gpu::mem_copy(dst, src, n);
    }
#endif
  }

  void* raw_allocate(size_t n) final
  {
    return reinterpret_cast<void*>(
      AllocatorType::allocate(n, h2::ComputeStream {D}));
  }

  void raw_deallocate(void* ptr) final
  {
    AllocatorType::deallocate(reinterpret_cast<std::byte*>(ptr),
                              h2::ComputeStream {D});
  }

  c10::Device get_device() const noexcept final
  {
    return c10::Device {LBANNDeviceT, D != h2::Device::CPU};
  }

  ///@}

  // Singleton
  static H2AllocatorWrapper& instance()
  {
    static H2AllocatorWrapper<D> allocator;
    return allocator;
  }

private:
  H2AllocatorWrapper() = default;
  ~H2AllocatorWrapper() = default;
  H2AllocatorWrapper(H2AllocatorWrapper const&) = delete;
  H2AllocatorWrapper(H2AllocatorWrapper&&) = delete;
  H2AllocatorWrapper& operator=(H2AllocatorWrapper const&) = delete;
  H2AllocatorWrapper& operator=(H2AllocatorWrapper&&) = delete;
};

using H2CPUAllocatorWrapper = H2AllocatorWrapper<h2::Device::CPU>;
#if LBANNV2_HAS_GPU
using H2GPUAllocatorWrapper = H2AllocatorWrapper<h2::Device::GPU>;
#endif

}  // namespace lbannv2
