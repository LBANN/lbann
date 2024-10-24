////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include <lbannv2_config.h>

#include <lbannv2/ops/scalar.hpp>
#include <lbannv2/utils/errors.hpp>

#include <ATen/ops/_local_scalar_dense_native.h>

#if LBANNV2_WITH_MI300A || LBANNV2_UNKNOWN_MI300A
#include <lbannv2/types.hpp>
#include <lbannv2/utils/logging.hpp>

#include <h2/gpu/runtime.hpp>

#include <ATen/core/TensorBase.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <c10/hip/HIPStream.h>

// FIXME: We should integrate this better with either H2 dispatch or
// Torch dispatch (I don't really care, honestly).
namespace
{

template <typename ScalarT>
at::Scalar mi300a_impl(at::Tensor const& self)
{
  // The contract is a sync, so we sync. (It's also likely a
  // requirement for correctness, so we can assume the value can be
  // safely accessed.
  auto const stream = at::hip::getCurrentHIPStream();
  h2::gpu::sync(stream);
  return at::Scalar(*reinterpret_cast<ScalarT const*>(self.const_data_ptr()));
}

at::Scalar mi300a_dispatch(at::Tensor const& self)
{
  c10::ScalarType const dtype = self.scalar_type();

  LBANNV2_TRACE("lbannv2::_local_scalar_dense_mi300a(device={}, dtype={})",
                self.device().str(),
                c10::toString(dtype));
  switch (dtype)
  {
  case c10::ScalarType::Bool: return mi300a_impl<bool>(self);
  case c10::ScalarType::Float: return mi300a_impl<float>(self);
  case c10::ScalarType::Double: return mi300a_impl<double>(self);
  case c10::ScalarType::Int: return mi300a_impl<int>(self);
  case c10::ScalarType::UInt32: return mi300a_impl<std::uint32_t>(self);
  case c10::ScalarType::Long: return mi300a_impl<long>(self);
  default: return at::native::_local_scalar_dense_cuda(self);
  }
}
}  // namespace
#endif // LBANNV2_WITH_MI300A || LBANNV2_UNKNOWN_MI300A

at::Scalar lbannv2::local_scalar_dense_hip(at::Tensor const& self)
{
  // self.numel() == 1 is asserted elsewhere.
  c10::ScalarType const dtype = self.dtype().toScalarType();

  // Technically, the "right" fallback is implemented in all
  // subsequent code paths, but I want to know about it if there's
  // another type we should be supporting.
  LBANNV2_ASSERT(is_supported(dtype), std::runtime_error, c10::toString(dtype));

#if LBANNV2_WITH_MI300A || LBANNV2_UNKNOWN_MI300A
#if LBANNV2_UNKNOWN_MI300A
  if (h2::gpu::is_integrated())
#endif  // LBANNV2_UNKNOWN_MI300A
    return mi300a_dispatch(self);
#endif  // LBANNV2_WITH_MI300A || LBANNV2_UNKNOWN_MI300A

  // Fallback to the Torch impl (cannot call at::_local_scalar_dense
  // -- it will cause an infinite recursion through this function).
  return at::native::_local_scalar_dense_cuda(self);
}
