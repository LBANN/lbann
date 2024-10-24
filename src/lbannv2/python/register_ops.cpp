////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include "lbannv2_config.h"

#include "lbannv2/backend/fallback.hpp"
#include <lbannv2/ops/copy.hpp>
#include <lbannv2/ops/empty_tensor.hpp>
#include <lbannv2/utils/device_helpers.hpp>
#include <lbannv2/utils/errors.hpp>
#include <lbannv2/utils/logging.hpp>

#if LBANNV2_WITH_MI300A || LBANNV2_UNKNOWN_MI300A
#include <lbannv2/memory/mi300a_allocator.hpp>
#include <lbannv2/ops/nonzero.hpp>
#include <lbannv2/ops/scalar.hpp>
#endif

#include <ATen/ATen.h>
#include <c10/core/Device.h>

#include <utility>

#include <c10/util/ArrayRef.h>
#include <torch/extension.h>
#include <torch/library.h>

// TODO: Look into c10::BackendMetadata as a way to include h2
//       information opaquely in the TensorImpl

namespace
{

at::Tensor lbannv2_empty_memory_format(
  c10::IntArrayRef size,
  ::std::optional<at::ScalarType> dtype_opt,
  ::std::optional<at::Layout> layout_opt,
  ::std::optional<at::Device> device_opt,
  ::std::optional<bool> pin_memory_opt,
  ::std::optional<c10::MemoryFormat> memory_format_opt)
{
  return lbannv2::empty_lbann(std::move(size),
                              std::move(dtype_opt),
                              std::move(layout_opt),
                              std::move(device_opt),
                              std::move(pin_memory_opt),
                              std::move(memory_format_opt));
}

at::Tensor lbannv2_empty_strided(c10::IntArrayRef size,
                                 c10::IntArrayRef stride,
                                 ::std::optional<at::ScalarType> dtype,
                                 ::std::optional<at::Layout> layout,
                                 ::std::optional<at::Device> device,
                                 ::std::optional<bool> pin_memory)
{
  return lbannv2::empty_strided_lbann(std::move(size),
                                      std::move(stride),
                                      std::move(dtype),
                                      std::move(layout),
                                      std::move(device),
                                      std::move(pin_memory));
}

at::Tensor lbannv2__copy_from(at::Tensor const& self,
                              at::Tensor const& dst,
                              bool non_blocking)
{
  return lbannv2::copy_from(self, dst, non_blocking);
}

}  // namespace

// Pass an explicit string!
#define EXPLICIT_LBANNV2_FALLBACK(kernel_name)                                 \
  m.impl(                                                                      \
    kernel_name,                                                               \
    torch::CppFunction::makeFromBoxedFunction<&lbannv2::lbannv2_fallback>())

TORCH_LIBRARY_IMPL(_, PrivateUse1, m)
{
  m.fallback(
    torch::CppFunction::makeFromBoxedFunction<&lbannv2::lbannv2_fallback>());
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
  m.impl("empty.memory_format", TORCH_FN(lbannv2_empty_memory_format));
  m.impl("empty_strided", TORCH_FN(lbannv2_empty_strided));
  m.impl("_copy_from", TORCH_FN(lbannv2__copy_from));

  // Because there's a default for this, we were dispatching through
  // that, which landed on "convolution_overrideable". Since we
  // don't have an implementation for *that*, and that just
  // continually falls back to its exception, we shim this in here.
  EXPLICIT_LBANNV2_FALLBACK("convolution");
  EXPLICIT_LBANNV2_FALLBACK("convolution_backward");
}

#if LBANNV2_WITH_MI300A || LBANNV2_UNKNOWN_MI300A

namespace
{

at::Scalar lbannv2__local_scalar_dense_cuda(at::Tensor const& self)
{
  return lbannv2::local_scalar_dense_hip(self);
}

at::Tensor lbannv2_nonzero(at::Tensor const& self)
{
  return lbannv2::nonzero(self);
}

at::Tensor& lbannv2_nonzero_out(at::Tensor const& self, at::Tensor& out)
{
  return lbannv2::nonzero_out(self, out);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, CUDA, m)
{
  m.impl("_local_scalar_dense", TORCH_FN(lbannv2__local_scalar_dense_cuda));
  m.impl("nonzero", TORCH_FN(lbannv2_nonzero));
  m.impl("nonzero.out", TORCH_FN(lbannv2_nonzero_out));
}

#endif // LBANNV2_WITH_MI300A || LBANNV2_UNKNOWN_MI300A
