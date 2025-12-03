////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include <lbannv2/memory/mi300a_allocator.hpp>
#include <lbannv2/memory/registry.hpp>

#include "test_helpers.hpp"

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
c10::Device lbann_gpu() noexcept
{
  return c10::Device {c10::kCUDA,
                      static_cast<c10::DeviceIndex>(h2::gpu::current_gpu())};
}
}  // namespace

TEST_CASE("MI300Allocator::allocate and MI300Allocator::deallocate",
          "[memory][mi300a]")
{
  SKIP_WHEN_NO_MI300A();

  auto& alloc = lbannv2::MI300Allocator::instance();
  size_t const size = 64;
  auto ptr = alloc.allocate(size);
  CHECK(ptr.device() == lbann_cpu());
  CHECK(lbannv2::pointer_registry().known(ptr.get()));
}

TEST_CASE("migrate_ptr CPU to GPU", "[memory][mi300a]")
{
  SKIP_WHEN_NO_MI300A();

  auto& alloc = lbannv2::MI300Allocator::instance();
  size_t const size = 64;
  auto ptr = alloc.allocate(size);

  // Migrate to GPU
  CHECK(lbannv2::pointer_registry().get_allocator(ptr.get_context()) == &alloc);
  CHECK_NOTHROW(lbannv2::migrate_ptr(
    ptr, lbann_gpu(), c10::Stream(c10::Stream::DEFAULT, lbann_gpu())));
  CHECK(ptr.device() == lbann_gpu());
  CHECK(lbannv2::pointer_registry().get_allocator(ptr.get_context())
        == &lbannv2::get_allocator(lbann_gpu()));
}

TEST_CASE("migrate_ptr GPU to CPU", "[memory][mi300a]")
{
  SKIP_WHEN_NO_MI300A();

  auto& alloc = lbannv2::get_allocator(lbann_gpu());
  size_t const size = 64;
  auto ptr = alloc.allocate(size);

  // Migrate to CPU
  CHECK(lbannv2::pointer_registry().get_allocator(ptr.get_context()) == &alloc);
  CHECK_NOTHROW(lbannv2::migrate_ptr(
    ptr, lbann_cpu(), c10::Stream(c10::Stream::DEFAULT, lbann_cpu())));
  CHECK(ptr.device() == lbann_cpu());
  CHECK(lbannv2::pointer_registry().get_allocator(ptr.get_context())
        == &lbannv2::get_allocator(lbann_cpu()));
}
