////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <lbannv2_config.h>

#include <c10/core/Allocator.h>

namespace lbannv2
{

/** @class Allocator
 *  @brief A simplistic interface for LBANN allocators.
 */
class LBANNV2_EXPORT Allocator : public c10::Allocator
{
public:
  virtual void* raw_alloc(size_t nbytes) = 0;
  virtual void raw_dealloc(void* ptr) = 0;
  virtual c10::Device get_device() const noexcept = 0;

  c10::DataPtr allocate(size_t n) final;
};  // class Allocator

LBANNV2_EXPORT bool is_managed_ptr(void const* ptr) noexcept;

LBANNV2_EXPORT void use_mi300a_cpu_allocator();
LBANNV2_EXPORT void use_torch_cpu_allocator();

}  // namespace lbannv2
