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
// c10::Allocator requires a function-pointer-compatible function to
// return from `raw_deleter()`. This is that function. Basically this
// calls `get_allocator().raw_delete(ptr)`.
LBANNV2_EXPORT void delete_managed_ptr(void* ptr);

LBANNV2_EXPORT bool is_managed_ptr(void const* ptr) noexcept;

/** @class Allocator
 *  @brief A simplistic interface for LBANN allocators.
 */
class LBANNV2_EXPORT Allocator : public c10::Allocator
{
public:
  virtual void* raw_allocate(size_t nbytes) = 0;
  virtual void raw_deallocate(void* ptr) = 0;
  virtual c10::Device get_device() const noexcept = 0;

  c10::DataPtr allocate(size_t n) final;
  c10::DeleterFnPtr raw_deleter() const noexcept final
  {
    return &delete_managed_ptr;
  }
};  // class Allocator

LBANNV2_EXPORT Allocator& get_allocator(c10::Device const& lbann_device,
                                        bool pinned = false);

LBANNV2_EXPORT Allocator& get_pinned_memory_allocator();

LBANNV2_EXPORT void set_allocator(c10::Device const& lbann_device,
                                  Allocator* alloc);

}  // namespace lbannv2
