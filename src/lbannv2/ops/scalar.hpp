////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <lbannv2_config.h>

#include <ATen/core/Tensor.h>

namespace lbannv2
{

LBANNV2_EXPORT at::Scalar local_scalar_dense_hip(at::Tensor const&);

}  // namespace lbannv2
