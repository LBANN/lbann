////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include "lbannv2_config.h"

#include "lbannv2/memory/mi300a_allocator.hpp"

#include "lbannv2/memory/registry.hpp"
#include "lbannv2/utils/errors.hpp"
#include "lbannv2/utils/gpu_utils.hpp"
#include "lbannv2/utils/logging.hpp"

#if LBANNV2_HAS_CUDA
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAStream.h>
#elif LBANNV2_HAS_ROCM
#include <ATen/hip/HIPContextLight.h>
#include <c10/hip/HIPStream.h>
#endif

#include <c10/core/CachingDeviceAllocator.h>

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
  ::lbannv2::TorchGPUStream_t stream;

  StreamRAII()
    : stream {lbannv2::c10_gpu::getStreamFromExternal(
        use_nonblocking_stream() ? lbannv2::gpu::make_nonblocking_stream()
                                 : lbannv2::gpu::make_stream(),
        lbannv2::gpu::current_device())}
  {}
  ~StreamRAII()
  {
    try
    {
      lbannv2::gpu::destroy_stream(stream.stream());
    }
    catch (...)
    {}
  }
};  // struct StreamRAII

// Internal stream for managing "host" allocations through CUB
::lbannv2::TorchGPUStream_t host_allocation_stream(c10::DeviceIndex const idx)
{
  static std::vector<StreamRAII> stream_raii(lbannv2::gpu::num_devices());
  LBANNV2_ASSERT_ALWAYS(idx >= 0 && idx < lbannv2::gpu::num_devices());
  return stream_raii[idx].stream;
}

c10::Device resolve_device(c10::Device const& d)
{
  if (d.is_cuda() && !d.has_index())
    return {c10::kCUDA, lbannv2::gpu::current_device()};

  return d;
}

#if LBANNV2_USE_C10_HIP_NAMESPACE_AND_SYMBOLS
namespace DeviceAlloc_ns = c10::hip::HIPCachingAllocator;
#else
namespace DeviceAlloc_ns = c10::cuda::CUDACachingAllocator;
#endif

void lbannv2_report_free(DeviceAlloc_ns::TraceEntry const& entry)
{
  try
  {
    void* const ptr = reinterpret_cast<void*>(entry.addr_);
    lbannv2::pointer_registry().remove(ptr);
    LBANNV2_TRACE("Deallocate (ptr={})", (void const*) ptr);
  }
  catch (lbannv2::UnknownAddress const&)
  {
    // ignore -- ptr allocated in Torch
  }
}

void lbannv2_trace_alloc(DeviceAlloc_ns::TraceEntry const& entry)
{
  if (entry.action_ == DeviceAlloc_ns::TraceEntry::FREE_COMPLETED)
    lbannv2_report_free(entry);
}
}  // namespace

namespace lbannv2
{

MI300Allocator::MI300Allocator()
{
#if LBANNV2_WITHOUT_MI300A || LBANNV2_UNKNOWN_MI300A
#if LBANNV2_UNKNOWN_MI300A
  if (!lbannv2::gpu::is_integrated())
#endif
    throw std::runtime_error("MI300Allocator is only supported on MI300A");
#endif

  auto* const dev_alloc =
    dynamic_cast<DeviceAlloc_t*>(at::cuda::getCUDADeviceAllocator());
  LBANNV2_ASSERT_ALWAYS(dev_alloc);
  if (!dev_alloc->initialized())
    dev_alloc->init(gpu::num_devices());

  // Trace memory stuff
  dev_alloc->attachAllocatorTraceTracker(lbannv2_trace_alloc);

  alloc_ = dev_alloc;
}

void MI300Allocator::copy_data(void* const dst,
                               void const* const src,
                               size_t const bytes) const
{
  LBANNV2_TRACE(
    "MI300Allocator::copy_data(dst={}, src={}, bytes={})", dst, src, bytes);
  std::memcpy(dst, src, bytes);
}

void* MI300Allocator::raw_alloc(size_t const nbytes)
{
  auto* const ptr = alloc_->raw_alloc_with_stream(
    nbytes, host_allocation_stream(lbannv2::gpu::current_device()));

  LBANNV2_TRACE(
    "MI300Allocator::raw_allocate(nbytes={}): ptr={}, current_device={}",
    nbytes,
    ptr,
    lbannv2::gpu::current_device());
  lbannv2::gpu::sync(host_allocation_stream(lbannv2::gpu::current_device()));

  return ptr;
}

void MI300Allocator::raw_dealloc(void* ptr)
{
  LBANNV2_TRACE("MI300Allocator::raw_deallocate(ptr={})", ptr);
  alloc_->raw_delete(ptr);
}

c10::Device MI300Allocator::get_device() const noexcept
{
  return c10::Device {c10::kCPU};
}

c10::DeleterFnPtr MI300Allocator::raw_deleter() const
{
  return alloc_->raw_deleter();
}

MI300Allocator& MI300Allocator::instance()
{
  static MI300Allocator alloc;
  return alloc;
}

}  // namespace lbannv2

c10::DeviceIndex lbannv2::get_device_idx(void const* const ptr) noexcept
{
  int device_idx;
  auto const hip_status = hipPointerGetAttribute(
    &device_idx, HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL, const_cast<void*>(ptr));
  if (hip_status == hipSuccess)
  {
    return static_cast<c10::DeviceIndex>(device_idx);
  }
  else
  {
    LBANNV2_DEBUG("lbannv2::get_device_idx(ptr={}) failed. Error: {}",
                  ptr,
                  hipGetErrorString(hip_status));
    return -1;
  }
}

// Let's aim for a fully robust implementation here. We must consider:
//   1. Migrating from D(:m) -> D(:m) is a no-op.
//   2. Migrating from D:m -> D:n is a deep copy
void lbannv2::migrate_ptr(c10::DataPtr& ptr,
                          c10::Device to_device,
                          c10::Stream with_stream)
{
  auto const real_tgt_device = resolve_device(to_device);

  // If no migration actually happens, just short-circuit...
  if (ptr.device() == real_tgt_device)
    return;

#if LBANNV2_WITHOUT_MI300A || LBANNV2_UNKNOWN_MI300A
#if LBANNV2_UNKNOWN_MI300A
  if (!lbannv2::gpu::is_integrated())
#endif
  {
    throw std::runtime_error("migrate_ptr is only supported on MI300A");
  }
#endif

  // Check that the migration is valid
  auto const ptr_dev_idx = get_device_idx(ptr.get_context());
  c10::Device const real_src_device = ptr_dev_idx == -1
                                        ? c10::Device {c10::kCPU}
                                        : c10::Device {c10::kCUDA, ptr_dev_idx};
  LBANNV2_ASSERT(real_tgt_device.is_cpu() || real_src_device == real_tgt_device,
                 std::runtime_error,
                 "lbannv2::migrate_ptr: invalid src/tgt device combo");

  // Update the stream
  auto const new_stream = real_tgt_device.is_cpu()
                            ? host_allocation_stream(ptr_dev_idx)
                            : TorchGPUStream_t(with_stream);

  // UGH. Oh well.
  MI300Allocator::instance().alloc_->recordStream(ptr, new_stream);

  // Finally, update the DataPtr itself
  ptr.unsafe_set_device(real_tgt_device);
}
