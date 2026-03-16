////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include <lbannv2/memory/mi300a_allocator.hpp>
#include <lbannv2/memory/registry.hpp>
#include <lbannv2/utils/gpu_utils.hpp>

#include "test_helpers.hpp"

#include <ATen/hip/HIPContextLight.h>
#include <c10/core/Allocator.h>
#include <c10/core/CPUAllocator.h>
#include <c10/hip/HIPCachingAllocator.h>
#include <c10/hip/HIPStream.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

namespace
{

void do_raw_allocate(void** ptr, size_t size, lbannv2::MI300Allocator& alloc)
{
  *ptr = alloc.raw_allocate(size);
}

}  // namespace

TEST_CASE("MI300Allocator::raw_allocate and MI300Allocator::raw_deallocate",
          "[memory][mi300a]")
{
  SKIP_WHEN_NO_MI300A();

  auto& alloc = lbannv2::MI300Allocator::instance();
  size_t const size = 64;
  void* ptr = nullptr;
  REQUIRE_NOTHROW(do_raw_allocate(&ptr, size, alloc));

  CHECK(ptr != nullptr);

  REQUIRE_NOTHROW(alloc.raw_deallocate(ptr));
}

namespace
{
c10::Device lbann_cpu() noexcept
{
  return c10::Device {c10::kCPU};
}
// c10::Device lbann_gpu() noexcept
// {
//   return c10::Device {
//     c10::kCUDA, static_cast<c10::DeviceIndex>(lbannv2::gpu::current_device())};
// }
}  // namespace

TEST_CASE("MI300Allocator::allocate and MI300Allocator::deallocate",
          "[memory][mi300a]")
{
  SKIP_WHEN_NO_MI300A();

  auto& alloc = lbannv2::MI300Allocator::instance();
  size_t const size = 64;

  void* raw_ptr = nullptr;
  {
    auto ptr = alloc.allocate(size);
    raw_ptr = ptr.get();
    CHECK(ptr.device() == lbann_cpu());
    CHECK(lbannv2::pointer_registry().known(raw_ptr));
  }

  // DataPtr goes out of scope, should be deleted.

  CHECK(!lbannv2::pointer_registry().known(raw_ptr));
}
