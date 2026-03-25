////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <lbannv2/memory/allocator.hpp>
#include <lbannv2/utils/gpu_utils.hpp>

#include <c10/core/Stream.h>

#if LBANNV2_HAS_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#elif LBANNV2_HAS_ROCM
#include <c10/hip/HIPCachingAllocator.h>
#endif

namespace lbannv2
{

// Call when moving pointer to a different device
void migrate_ptr(c10::DataPtr& ptr,
                 c10::Device to_device,
                 c10::Stream with_stream);

class MI300Allocator final : public Allocator
{
public:
  void copy_data(void* dst, void const* src, size_t bytes) const final;

  void* raw_allocate(size_t nbytes) final;

  void raw_deallocate(void* ptr) final;

  c10::DeleterFnPtr raw_deleter() const final;

  c10::Device get_device() const noexcept final;

  static MI300Allocator& instance();

private:
  MI300Allocator();
  ~MI300Allocator() = default;
  MI300Allocator(MI300Allocator const&) = delete;
  MI300Allocator(MI300Allocator&&) = delete;
  MI300Allocator& operator=(MI300Allocator const&) = delete;
  MI300Allocator& operator=(MI300Allocator&&) = delete;

#if LBANNV2_USE_C10_HIP_NAMESPACE_AND_SYMBOLS
  using DeviceAlloc_t = ::c10::hip::HIPCachingAllocator::HIPAllocator;
#else
  using DeviceAlloc_t = ::c10::cuda::CUDACachingAllocator::CUDAAllocator;
#endif
  DeviceAlloc_t* alloc_;

  friend void migrate_ptr(c10::DataPtr&, c10::Device, c10::Stream);

};

/** @brief Get the device with which the allocation is associated.
 *
 * @note From what I can tell, this is any valid pointer -- it doesn't
 *       have to be the "context" pointer, for instance.
 *
 * @param[in] A pointer to valid memory.
 *
 * @returns The (GPU) device index with which the allocation is
 *          associated. -1 if not GPU memory or nullptr.
 */
c10::DeviceIndex get_device_idx(void const* const ptr) noexcept;

}  // namespace lbannv2
