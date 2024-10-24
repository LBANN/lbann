////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <ATen/ATen.h>

namespace lbannv2
{

at::Tensor nonzero(at::Tensor const& self);
at::Tensor& nonzero_out(at::Tensor const& self, at::Tensor& out);

} // namespace lbannv2
