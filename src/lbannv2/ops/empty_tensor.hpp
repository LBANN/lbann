////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <lbannv2_export.h>

#include <ATen/core/TensorBase.h>
#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>

namespace lbannv2
{

LBANNV2_EXPORT at::TensorBase empty_lbann(c10::IntArrayRef size,
                                          c10::TensorOptions const& options);

LBANNV2_EXPORT at::TensorBase
empty_lbann(c10::IntArrayRef size,
            std::optional<c10::ScalarType> dtype_opt,
            std::optional<c10::Layout> layout_opt,
            std::optional<c10::Device> device_opt,
            std::optional<bool> pin_memory_opt,
            std::optional<c10::MemoryFormat> memory_format_opt);

LBANNV2_EXPORT at::TensorBase
empty_strided_lbann(c10::IntArrayRef size,
                    c10::IntArrayRef stride,
                    c10::TensorOptions const& options);

LBANNV2_EXPORT at::TensorBase
empty_strided_lbann(c10::IntArrayRef size,
                    c10::IntArrayRef stride,
                    std::optional<c10::ScalarType> dtype_opt,
                    std::optional<c10::Layout> layout_opt,
                    std::optional<c10::Device> device_opt,
                    std::optional<bool> pin_memory_opt);

}  // namespace lbannv2
