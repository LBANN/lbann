////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include "lbannv2/memory/allocator.hpp"

#include "lbannv2/memory/h2_allocator_wrappers.hpp"
#include "lbannv2/memory/registry.hpp"
#include "lbannv2/utils/device_helpers.hpp"
#include "lbannv2/utils/errors.hpp"
#include "lbannv2/utils/logging.hpp"

#if LBANNV2_WITH_MI300A || LBANNV2_UNKNOWN_MI300A
#include "lbannv2/memory/mi300a_allocator.hpp"
#endif

namespace lbannv2
{

c10::DataPtr Allocator::allocate(size_t n)
{
  // Do the allocation
  void* const buffer = raw_allocate(n);

  // Log the allocation
  LBANNV2_TRACE("Allocator::allocate(n={}, ptr={})", n, buffer);
  pointer_registry().add(buffer, n, this);

  // Decorate the allocation.
  return {buffer, buffer, this->raw_deleter(), this->get_device()};
}

}  // namespace lbannv2

namespace
{
using alloc_map_type =
  std::array<::lbannv2::Allocator*, ::lbannv2::NumLBANNDevices>;

lbannv2::Allocator& get_cpu_allocator()
{
#if LBANNV2_WITH_MI300A
  return lbannv2::MI300Allocator::instance();
#elif LBANNV2_WITHOUT_MI300A
  return lbannv2::H2CPUAllocatorWrapper::instance();
#elif LBANNV2_UNKNOWN_MI300A
  if (h2::gpu::is_integrated())
    return lbannv2::MI300Allocator::instance();
  else
    return lbannv2::H2CPUAllocatorWrapper::instance();
#endif
}

alloc_map_type make_default_alloc_map()
{
  alloc_map_type map = {
    &get_cpu_allocator(),
#if LBANNV2_HAS_GPU
    &::lbannv2::H2GPUAllocatorWrapper::instance(),
#endif
  };
  return map;
}

alloc_map_type& alloc_map()
{
  static alloc_map_type allocators = make_default_alloc_map();
  return allocators;
}
}  // namespace

lbannv2::Allocator& lbannv2::get_pinned_memory_allocator()
{
  LBANNV2_WARN("No pinned allocator exposed yet; using regular CPU allocator.");
  return get_allocator(c10::Device {LBANNDeviceT, LBANN_CPU});
}

lbannv2::Allocator& lbannv2::get_allocator(c10::Device const& lbann_device,
                                           bool pinned)
{
  LBANNV2_ASSERT_ALWAYS(is_lbann(lbann_device));
  if (pinned)
  {
    LBANNV2_ASSERT_ALWAYS(lbann_device.index() == 0);
    return get_pinned_memory_allocator();
  }

  auto const index = lbann_device.index();

  LBANNV2_ASSERT_ALWAYS(0 <= index && index < NumLBANNDevices);
  auto* const alloc = alloc_map().at(index);

  LBANNV2_ASSERT_ALWAYS(static_cast<bool>(alloc));
  return *alloc;
}

void lbannv2::set_allocator(c10::Device const& lbann_device,
                            Allocator* const alloc)
{
  LBANNV2_ASSERT_ALWAYS(is_lbann(lbann_device));
  alloc_map().at(lbann_device.index()) = alloc;
}

void lbannv2::delete_managed_ptr(void* const ptr)
{
  LBANNV2_TRACE("delete_managed_ptr(ptr={})", ptr);
  try
  {
    ::lbannv2::Allocator* alloc = pointer_registry().get_allocator(ptr);
    LBANNV2_ASSERT_DEBUG(ptr == pointer_registry().get_context(ptr));
    alloc->raw_deallocate(ptr);
    pointer_registry().remove(ptr);
  }
  catch (UnknownAddress const&)
  {
    throw std::runtime_error("Ptr not allocated by this allocator.");
  }
  catch (std::runtime_error const&)
  {}
}

bool lbannv2::is_managed_ptr(void const* const ptr) noexcept
{
  return pointer_registry().known(ptr);
}
