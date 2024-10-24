////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include "lbannv2/memory/toplevel_allocator.hpp"

#include "lbannv2/backend/library_state.hpp"
#include "lbannv2/memory/h2_allocator_wrappers.hpp"
#include "lbannv2/memory/registry.hpp"
#include "lbannv2/types.hpp"
#include "lbannv2/utils/device_helpers.hpp"
#include "lbannv2/utils/errors.hpp"
#include "lbannv2/utils/logging.hpp"

#include <h2/core/allocator.hpp>

namespace lbannv2
{

GlobalAllocator::GlobalAllocator()
{}

GlobalAllocator::~GlobalAllocator()
{}

c10::DataPtr GlobalAllocator::allocate(size_t n)
{
  auto const lbann_dev = state::current_device_lbann();
  auto& dev_alloc = get_allocator(lbann_dev);

  // Get the actual allocation. This buffer will have the deleter of
  // device allocator attached to it to save an unnecessary lookup at
  // deletion time.
  return dev_alloc.allocate(n);

  // FIXME (trb): Do we want any logging here? It would be mostly
  // redundant to whatever the device allocator would log, but it
  // might be nice to know that it was allocated through this class
  // instead of directly? IDK. The utility of that might diminish as
  // we learn more about this ecosystem.
}

c10::DeleterFnPtr GlobalAllocator::raw_deleter() const
{
  // This looks up the pointer in the registry and deletes the pointer
  // through its associated allocator.
  return &lbannv2::delete_managed_ptr;
}

// FIXME (trb): Currently, this requires that src and dst be from the
// same underlying allocator. This could be relaxed in a few ways. It
// could be relaxed to allow different allocators for the same device
// (e.g., copying pinned cpu memory to non-pinned cpu memory). It
// could also be relaxed to allow inter-device-but-within-LBANN-memory
// copies. It's unclear how often this will actually be used, though,
// and this was easiest to implement sanely.
void GlobalAllocator::copy_data(void* const dst,
                                void const* const src,
                                size_t length) const
{
  LBANNV2_TRACE(
    "GlobalAllocator::copy_data(dst={}, src={}, length={})", dst, src, length);

  try
  {
    ::lbannv2::Allocator const* const src_alloc =
      pointer_registry().get_allocator(src);
    ::lbannv2::Allocator const* const dest_alloc =
      pointer_registry().get_allocator(dst);
    LBANNV2_ASSERT_ALWAYS(src_alloc == dest_alloc);
    dest_alloc->copy_data(dst, src, length);
  }
  catch (UnknownAddress const&)
  {
    throw std::runtime_error(
      "At least one of the pointers was not allocated by LBANN allocator");
  }
  // FIXME (trb): We could be more aggressive about checking that the
  // full ranges `[src, src+length)` and `[dst, dst+length)` are
  // valid, perhaps in debug mode?
}

GlobalAllocator& GlobalAllocator::instance()
{
  static GlobalAllocator allocator;
  return allocator;
}

}  // namespace lbannv2

auto lbannv2::get_allocator() -> GlobalAllocator&
{
  return GlobalAllocator::instance();
}

REGISTER_ALLOCATOR(lbannv2::LBANNDeviceT, &::lbannv2::get_allocator());
