////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include "lbannv2_config.h"

#include "lbannv2/memory/mi300a_allocator.hpp"

#include "lbannv2/backend/library_state.hpp"
#include "lbannv2/memory/registry.hpp"
#include "lbannv2/utils/errors.hpp"
#include "lbannv2/utils/logging.hpp"

#include <h2/core/sync.hpp>

#include <c10/hip/HIPStream.h>

namespace
{
bool get_use_nonblocking_stream_env_var()
{
  char* env = std::getenv("LBANNV2_NONBLOCKING_HOST_ALLOC_STREAM");
  return env && std::strlen(env) && env[0] != '0';
}

bool use_nonblocking_stream()
{
  static bool const nonblock = get_use_nonblocking_stream_env_var();
  LBANNV2_DEBUG("Using nonblocking MI300A allocation stream? {}", nonblock);
  return nonblock;
}

struct StreamRAII
{
  h2::gpu::DeviceStream stream;
  StreamRAII()
    : stream {use_nonblocking_stream() ? h2::gpu::make_stream_nonblocking()
                                       : h2::gpu::make_stream()} {};
  ~StreamRAII()
  {
    try
    {
      h2::gpu::destroy(stream);
    }
    catch (...)
    {}
  }
};  // struct StreamRAII

// Internal stream for managing "host" allocations through CUB
h2::gpu::DeviceStream host_allocation_stream()
{
  static StreamRAII stream_raii;
  return stream_raii.stream;
}

// FIXME: Implement this more robustly (probably requires LBANN
// backend streams to be fleshed out, see how CUDA does this, e.g.).
h2::gpu::DeviceStream get_raw_stream(c10::Stream const stream)
{
  return c10::hip::getCurrentHIPStream();
}

}  // namespace

namespace lbannv2
{

MI300Allocator::MI300Allocator()
{
#if LBANNV2_WITHOUT_MI300A || LBANNV2_UNKNOWN_MI300A
#if LBANNV2_UNKNOWN_MI300A
  if (!h2::gpu::is_integrated())
#endif
    throw std::runtime_error("MI300Allocator is only supported on MI300A");
#endif
}

void MI300Allocator::copy_data(void* const dst,
                               void const* const src,
                               size_t const bytes) const
{
  LBANNV2_TRACE(
    "MI300Allocator::copy_data(dst={}, src={}, bytes={})", dst, src, bytes);
  std::memcpy(dst, src, bytes);
}

void* MI300Allocator::raw_allocate(size_t const nbytes)
{
  void* ptr;
  H2_CHECK_HIP(h2::gpu::default_cub_allocator().DeviceAllocate(
    &ptr, nbytes, host_allocation_stream()));
  h2::gpu::sync(host_allocation_stream());
  return ptr;
}

void MI300Allocator::raw_deallocate(void* ptr)
{
  H2_CHECK_HIP(h2::gpu::default_cub_allocator().DeviceFree(ptr));
}

c10::Device MI300Allocator::get_device() const noexcept
{
  return c10::Device {LBANNDeviceT, LBANN_CPU};
}

MI300Allocator& MI300Allocator::instance()
{
  static MI300Allocator alloc;
  return alloc;
}

}  // namespace lbannv2

namespace
{
bool is_ok_device(c10::Device const& dev)
{
  return lbannv2::is_lbann(dev) || (dev.type() == c10::kCPU)
         || (dev.type() == c10::kCUDA);
}

bool is_cpu_device(c10::Device const& dev)
{
  return (lbannv2::is_lbann(dev) && dev.index() == lbannv2::LBANN_CPU)
         || (dev.type() == c10::kCPU);
}

}  // namespace

void lbannv2::migrate_ptr(c10::DataPtr& ptr,
                          c10::Device to_device,
                          c10::Stream with_stream)
{
  // Maybe a bit too permissive here, but let's be nice.
  //
  // If no migration actually happens, just short-circuit...
  if (ptr.device() == to_device)
    return;

#if LBANNV2_WITHOUT_MI300A || LBANNV2_UNKNOWN_MI300A
#if LBANNV2_UNKNOWN_MI300A
  if (!h2::gpu::is_integrated())
#endif
    throw std::runtime_error("migrate_ptr is only supported on MI300A");
#endif

  // We can support any pointer from any backend that we have
  // allocated using our CUB allocator.

  auto& ptr_registry = pointer_registry();
  LBANNV2_ASSERT(is_ok_device(ptr.device()) && is_ok_device(to_device),
                 std::runtime_error,
                 "Migrate: unsupported device");

  // Find the live block in the cub allocator and replace the stream
  using BlkDesc = typename h2::gpu::RawCUBAllocType::BlockDescriptor;
  auto& cub_alloc = h2::gpu::default_cub_allocator();
  auto new_stream = is_cpu_device(to_device) ? host_allocation_stream()
                                             : get_raw_stream(with_stream);

  {
    std::lock_guard<std::mutex> lock {cub_alloc.mutex};
    BlkDesc key {ptr.get_context(), lbannv2::state::gpu_idx()};

    // Check that we only have one matching block
    LBANNV2_ASSERT(cub_alloc.live_blocks.count(key) == 1,
                   std::runtime_error,
                   "Migrate: pointer not managed by CUB!");

    // Must be found because count(key) == 1.
    auto blk_itr = cub_alloc.live_blocks.find(key);

    // Iterators to keys are always const. However, the comparison
    // function for "live_blocks" only compares ptr addrs and dev ids;
    // it does NOT compare the streams. So we commit the following
    // atrocity against, technically, the standard library:
    BlkDesc& blk = const_cast<BlkDesc&>(*blk_itr);
    blk.associated_stream = new_stream;
    // The other (equivalent?) option would be to create a copy of the
    // BlkDesc, set the proper stream, and then replace the block in
    // the set. But that seems like too much work.
    //
    // It's annoying that the implementation uses multisets for both
    // live and cached blocks (cached clearly makes sense -- it
    // compares based on size, and there can be many allocs with the
    // same size -- but I'm less clear on why live blocks are managed
    // with one). I currently assert that the pointer is only
    // registered once, but if that fails, we can replace this with a
    // loop over 'equal_range()`.
  }

  // Update our internal bookkeeping
  Allocator& new_allocator = get_allocator(to_lbann(to_device));
  ptr_registry.unsafe_reset_allocator(ptr.get_context(), &new_allocator);

  // Finally, update the DataPtr itself
  ptr.unsafe_set_device(to_device);
}
