////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include <lbannv2_config.h>

#include <lbannv2/backend/library_state.hpp>
#include <lbannv2/memory/mi300a_allocator.hpp>
#include <lbannv2/memory/registry.hpp>
#include <lbannv2/ops/migrate.hpp>
#include <lbannv2/utils/device_helpers.hpp>
#include <lbannv2/utils/logging.hpp>
#include <lbannv2/utils/tensor_helpers.hpp>

#include <ATen/Tensor.h>
#include <c10/core/Device.h>

namespace
{

bool is_ok_device(c10::Device const& d)
{
  // A device is ok if it is an LBANN device OR one of the native PyTorch
  // devices we support.
  return lbannv2::is_lbann(d) || (d.type() == c10::kCPU)
#if LBANNV2_HAS_GPU
         || (d.type() == c10::kCUDA
             && (!d.has_index() || d.index() == lbannv2::state::gpu_idx()))
#endif
    ;
}

}  // namespace

at::Tensor lbannv2::migrate(at::Tensor& t, c10::Device const& d)
{
  auto const src_d = t.device();

  LBANNV2_TRACE(
    "migrate(ptr={}, from={}, to={})", t.data_ptr(), src_d.str(), d.str());

  LBANNV2_ASSERT(pointer_registry().known(t.const_data_ptr()),
                 std::runtime_error,
                 "Trying to migrate unknown ptr.");

  // FIXME (trb): Should this case invalidate t? (probably...)
  if (src_d == d)
    return t;

  LBANNV2_ASSERT(
    is_ok_device(src_d),
    std::runtime_error,
    "Migrate: source tensor's device type not supported by LBANNv2.");
  LBANNV2_ASSERT(is_ok_device(d),
                 std::runtime_error,
                 "Migrate: destination device type not supported by LBANNv2.");

#if LBANNV2_WITHOUT_MI300A
  return t.to(t.options().device(device),
              /*non_blocking=*/false,
              /*copy=*/false,
              /*memory_format=*/std::nullopt);
#else
#if LBANNV2_UNKNOWN_MI300A
  if (h2::gpu::is_integrated())
  {
#endif

    lbannv2::migrate_ptr(
      t.storage().mutable_data_ptr(), d, c10::Stream {c10::Stream::DEFAULT, d});

    auto storage = t.storage();
    auto out = at::detail::make_tensor<at::TensorImpl>(at::TensorImpl::VIEW,
                                                       std::move(storage),
                                                       get_default_keyset(d),
                                                       t.dtype());
    sync_metadata(t, out);

    return out;
#if LBANNV2_UNKNOWN_MI300A
  }
  else
    return t.to(t.options().device(device),
                /*non_blocking=*/false,
                /*copy=*/false,
                /*memory_format=*/std::nullopt);
#endif
#endif
}
