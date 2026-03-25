////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include <lbannv2_config.h>

#include <lbannv2/memory/mi300a_allocator.hpp>
#include <lbannv2/memory/registry.hpp>
#include <lbannv2/ops/migrate.hpp>
#include <lbannv2/utils/gpu_utils.hpp>
#include <lbannv2/utils/logging.hpp>
#include <lbannv2/utils/tensor_helpers.hpp>

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#if LBANNV2_HAS_ROCM
#include <c10/hip/HIPFunctions.h>
#endif

#if LBANNV2_WITH_MI300A || LBANNV2_UNKNOWN_MI300A
namespace
{

// NOTE: This function assumes a binary view of memory: pointers only
// come from "CPU" or "CUDA" (i.e., HIP).
at::Device get_origin_device(void const* const ptr)
{
  // Note to future!me: the HIP runtime can give us both the context
  // pointer and the buffer size for any pointer allocated by HIP.
  // HOWEVER, so can the pytorch DataPtr object, which we have in the
  // context in which this function is used...
  int device_idx;
  if (hipPointerGetAttribute(&device_idx,
                             HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                             const_cast<void*>(ptr))
      == hipSuccess)
  {
    return {c10::kCUDA, static_cast<c10::DeviceIndex>(device_idx)};
  }
  return c10::kCPU;
}

// PyTorch still admits the possibility of a single process using
// multiple GPUs, though this historically has not been LBANN's
// preferred approach (instead preferring 1 GPU per rank and 1 rank
// per GPU). On MI300A, we can migrate a pointer from any "GPU" to the
// CPU freely. HOWEVER, we can only migrate from the CPU to the
// specific device on which the migrateable memory was allocated.
bool is_ok_device(c10::Device const& d)
{
  return d.is_cpu()
#if LBANNV2_HAS_GPU
         || d.is_cuda()
#endif
    ;
}

c10::DispatchKeySet get_default_keyset(c10::Device const& d)
{
  switch (d.type())
  {
  case c10::kCPU: return c10::DispatchKeySet {c10::DispatchKey::CPU};
  case c10::kCUDA: return c10::DispatchKeySet {c10::DispatchKey::CUDA};
  default: throw std::runtime_error("Unknown device type");
  }
}

}  // namespace
#endif

at::Tensor lbannv2::migrate(at::Tensor& t, c10::Device const& d)
{
  auto const src_d = t.device();
  LBANNV2_TRACE(
    "migrate(ptr={}, from={}, to={})", t.data_ptr(), src_d.str(), d.str());

  // Short-circuit
  if (src_d == d)
    return t;

#if LBANNV2_UNKNOWN_MI300A || LBANNV2_WITH_MI300A
  // NOTE: "LBANNV2_HAS_ROCM" is implied here.

  // At its heart, this isn't really "migrate", it's "rebrand"... I
  // don't actually care what the device annotations on the Tensor or
  // Storage are, I care about the origin of the pointer. It might
  // also be good to look into p2p memory access, but I don't know how
  // to query that just given the pointer (i.e., even if p2p mem
  // access is enabled *now*, I haven't discovered a way to tell if it
  // was enabled when a particular buffer was allocated (well, other
  // than trying to read it and letting the segfault happen)).
  auto const real_src_d = get_origin_device(t.const_data_ptr());

  // We need to get the "real" "CUDA" target.
  c10::Device const real_tgt_d =
    (d.is_cuda() && !d.has_index())
      ? c10::Device {c10::kCUDA, gpu::current_device()}
      : d;

  // If the real_src_d is "cpu", it can be migrated to "cpu".
  // If the real_src_d is "cuda:N", it can be migrated to "cpu" or "cuda:N".
  LBANNV2_ASSERT(real_tgt_d.is_cpu() || (real_src_d == real_tgt_d),
                 std::runtime_error,
                 "Migrate: ptr is not migrateable to given device.");
  LBANNV2_ASSERT(
    is_ok_device(real_src_d),
    std::runtime_error,
    "Migrate: source tensor's device type not supported by LBANNv2.");
  LBANNV2_ASSERT(is_ok_device(real_tgt_d),
                 std::runtime_error,
                 "Migrate: destination device type not supported by LBANNv2.");

  // FIXME: If the pointer is not owned by LBANNv2, how do we handle
  // its associated stream?
  //  ---> The PyTorch CUDA caching allocator provides "recordStream" :)

#if LBANNV2_UNKNOWN_MI300A
  if (lbannv2::gpu::is_integrated())
#endif
  {
    c10::Stream stream = real_tgt_d.is_cpu()
                           ? c10::Stream {c10::Stream::DEFAULT, d}
                           : getDeviceCurrentStream(real_tgt_d.index());

    lbannv2::migrate_ptr(t.storage().mutable_data_ptr(), d, stream);

    // Report the number of meaningful bytes migrated. This is
    // inherently based on the tensor shape rather than the allocated
    // buffer size (think: binned allocations, subtensor "views",
    // etc).
    LBANNV2_TRACE("migrated {} bytes (ptr={})",
                  std::accumulate(t.sizes().cbegin(),
                                  t.sizes().cend(),
                                  static_cast<int64_t>(1),
                                  std::multiplies<int64_t> {})
                    * t.dtype().itemsize(),
                  t.const_data_ptr());

    auto storage = t.storage();
    // FIXME (trb): I initially created this as a 'VIEW', but that
    // puts it in "inference mode" (i.e., out.is_inference() == true).
    // This is bad for training workloads. We may need to be a bit
    // more careful in general here... E.g., migrating views to views,
    // etc.
    auto out =
      at::detail::make_tensor<at::TensorImpl>(  // at::TensorImpl::VIEW,
        std::move(storage),
        get_default_keyset(d),
        t.dtype());
    sync_metadata(t, out);

    if (src_d.is_cuda())
    {
      getDeviceCurrentStream(src_d.index()).synchronize();
    }

    return out;
  }
#endif
  return t.to(t.options().device(d));
}
