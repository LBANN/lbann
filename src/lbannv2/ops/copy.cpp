////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include "copy.hpp"

#include <lbannv2/utils/logging.hpp>
#include <lbannv2/utils/tensor_helpers.hpp>

#include <h2/utils/Error.hpp>

#include <ATen/Tensor.h>

// aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) ->
// Tensor(a!)
at::Tensor&
lbannv2::copy(at::Tensor& dst_in, at::Tensor const& src_in, bool non_blocking)
{
  LBANNV2_TRACE("copy(dst={}, src={}, nonblocking={})",
                to_str(dst_in),
                to_str(src_in),
                non_blocking);

  // Let's use ATen's copy kernels for now. We can think about whether
  // to use H2's instead. In the end, we need to do any of: (a) alias
  // everything into ATen and use those kernels, (b) alias everything
  // into H2 and use those kernels, or (c) implement native copy
  // kernels here.

  // Alias stuff.
  at::Tensor dst = alias_as_native_device(dst_in);
  at::Tensor src = alias_as_native_device(src_in);

  // Try the copy with aliased tensors.
  dst.copy_(src, non_blocking);

  // Restore {dst,src}_in's storage to their original values.
  sync_data_ptr_device(dst_in);
  sync_data_ptr_device(src_in);

  return dst_in;
}

// aten::_copy_from(Tensor self, Tensor dst, bool non_blocking=False) -> Tensor
at::Tensor lbannv2::copy_from(at::Tensor const& self,
                              at::Tensor const& dst,
                              bool non_blocking)
{
  LBANNV2_TRACE("copy_from(self={}, dst={}, nonblocking={})",
                to_str(self),
                to_str(dst),
                non_blocking);
  // Semantics gleaned from the MPS impl at
  // <pytorch>/aten/src/ATen/native/mps/operations/Copy.mm.
  return copy(const_cast<at::Tensor&>(dst), self, non_blocking);
}
