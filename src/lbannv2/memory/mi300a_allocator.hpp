////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <lbannv2/memory/allocator.hpp>
#include <lbannv2/utils/device_helpers.hpp>

#include <h2/gpu/memory_utils.hpp>
#include <h2/gpu/runtime.hpp>

#include <c10/core/Stream.h>

namespace lbannv2
{

// Call when moving pointer to a different device
void migrate_ptr(c10::DataPtr& ptr,
                 c10::Device to_device,
                 c10::Stream with_stream);

class MI300Allocator final : public Allocator
{
public:
  void copy_data(void* dst, void const* src, size_t bytes) const final;

  void* raw_allocate(size_t nbytes) final;

  void raw_deallocate(void* ptr) final;

  c10::Device get_device() const noexcept final;

  static MI300Allocator& instance();

private:
  MI300Allocator();
  ~MI300Allocator() = default;
  MI300Allocator(MI300Allocator const&) = delete;
  MI300Allocator(MI300Allocator&&) = delete;
  MI300Allocator& operator=(MI300Allocator const&) = delete;
  MI300Allocator& operator=(MI300Allocator&&) = delete;
};
}  // namespace lbannv2
