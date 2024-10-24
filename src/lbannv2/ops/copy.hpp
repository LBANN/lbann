////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <ATen/Tensor.h>

namespace lbannv2
{

at::Tensor&
copy(at::Tensor& dst_in, at::Tensor const& src_in, bool non_blocking);

// aten::_copy_from(Tensor self, Tensor dst, bool non_blocking=False) -> Tensor
at::Tensor
copy_from(at::Tensor const& self, at::Tensor const& dst, bool non_blocking);

}  // namespace lbannv2
