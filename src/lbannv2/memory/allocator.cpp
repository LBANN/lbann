////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include "lbannv2/memory/allocator.hpp"

#include "lbannv2/memory/registry.hpp"
#include "lbannv2/utils/errors.hpp"
#include "lbannv2/utils/logging.hpp"

#include <c10/core/CPUAllocator.h>

#if LBANNV2_HAS_CUDA
#include <ATen/cuda/CUDAContextLight.h>
#elif LBANNV2_HAS_ROCM
#include <ATen/hip/HIPContextLight.h>
#endif

#if LBANNV2_WITH_MI300A || LBANNV2_UNKNOWN_MI300A
#include "lbannv2/memory/mi300a_allocator.hpp"
#endif

namespace lbannv2
{

c10::DataPtr Allocator::allocate(size_t n)
{
  // Do the allocation
  void* const buffer = this->raw_alloc(n);

  // Log the allocation
  LBANNV2_TRACE("Allocator::allocate(n={}, ptr={})", n, buffer);
  //pointer_registry().add(buffer, n, this);

  // Decorate the allocation.
  return {buffer, buffer, this->raw_deleter(), this->get_device()};
}

}  // namespace lbannv2

bool lbannv2::is_managed_ptr(void const* const ptr) noexcept
{
  return pointer_registry().known(ptr);
}

namespace
{

c10::Allocator* pt_orig_cpu_alloc_ = nullptr;

}  // namespace

void lbannv2::use_mi300a_cpu_allocator()
{
#if LBANNV2_WITH_MI300A || LBANNV2_UNKNOWN_MI300A
#if LBANNV2_UNKNOWN_MI300A
  if (gpu::is_integrated())
#endif
  {
    if (!pt_orig_cpu_alloc_)
      pt_orig_cpu_alloc_ = c10::GetCPUAllocator();
    c10::SetCPUAllocator(&MI300Allocator::instance());
    return;
  }
#endif
  LBANNV2_WARN("No MI300A allocator available");
}

void lbannv2::use_torch_cpu_allocator()
{
  if (pt_orig_cpu_alloc_)
    c10::SetCPUAllocator(pt_orig_cpu_alloc_);
}
