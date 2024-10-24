////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <lbannv2_export.h>

#include <lbannv2/memory/allocator.hpp>
#include <lbannv2/utils/device_helpers.hpp>

#include <c10/core/Allocator.h>

namespace lbannv2
{

/** @class GlobalAllocator
 *  @brief A unified interface for LBANNv2 allocators.
 *  @todo GlobalAllocator is a dumb name.
 *  @todo We need to handle streams, etc.
 *
 *  @todo PyTorch does not seem to expose data type information to
 *        allocators, so we might want to think about how we want to
 *        handle pointer alignment issues.
 *
 *  The instance() of this should be the "top-level" allocator that is
 *  registered with PyTorch for the LBANN backend. It's somewhat
 *  unclear to me what benefit that gives us, but c'est la vie.
 *
 *  Allocations are dispatched downstream based on the current device.
 *  A downstream allocation is handled by other allocators that
 *  operate on raw (hardware-specific) memory. This should only be
 *  used if a more specific allocator cannot be acquired for whatever
 *  reason.
 */
class LBANNV2_EXPORT GlobalAllocator final : public c10::Allocator
{
public:
  /** @name Virtual function overrides */
  ///@{

  // Actually allocate the memory
  c10::DataPtr allocate(size_t n) final;

  // Return anything non-null here.
  c10::DeleterFnPtr raw_deleter() const final;

  // memcpy
  void copy_data(void* dest, void const* src, size_t length) const final;

  ///@}

  static GlobalAllocator& instance();

private:
  /** @name Lifetime management */
  ///@{
  GlobalAllocator();
  ~GlobalAllocator();
  GlobalAllocator(GlobalAllocator const&) = delete;
  GlobalAllocator(GlobalAllocator&&) = delete;
  GlobalAllocator& operator=(GlobalAllocator const&) = delete;
  GlobalAllocator& operator=(GlobalAllocator&&) = delete;
  ///@}

};  // class GlobalAllocator

LBANNV2_EXPORT GlobalAllocator& get_allocator();

}  // namespace lbannv2
