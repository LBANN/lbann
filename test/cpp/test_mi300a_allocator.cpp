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
c10::Device lbann_gpu() noexcept
{
  return c10::Device {
    c10::kCUDA, static_cast<c10::DeviceIndex>(lbannv2::gpu::current_device())};
}
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

// The "kernel" here is loosely inspired by Aluminum's "GPUWait", but
// less fussy about things like "cache-line allocation" and
// "atomics"... All I need is something to guarantee the stream isn't
// synced before the second allocation, and this saves me the trouble
// of compiling a HIP kernel.
TEST_CASE("MI300Allocator stream semantics are working", "[memory][mi300a]")
{
  auto const gpu = lbann_gpu();

  // Some memory we can use later.
  int32_t* wait_mem;
  LBANNV2_CHECK_GPU(hipMalloc(&wait_mem, sizeof(int32_t)));
  *wait_mem = 0;

  int32_t const wait_value = 1;
  auto& alloc = lbannv2::MI300Allocator::instance();
  size_t const size = 64;

  // open block
  //   do an allocation
  //   migrate allocation to GPU
  //   "run a kernel" on the same stream
  // close block (delete the allocation)
  // allocate new buffer
  // check old and new buffers have different addresses

  auto torch_stream = lbannv2::getDeviceCurrentStream(gpu.index());
  void* orig_ptr = nullptr;  // never dereferenced
  {
    auto ptr = alloc.allocate(size);
    // cache the buffer address -- NEVER DEREFERENCED
    orig_ptr = ptr.get();

    // Add the ptr to the stream on GPU
    lbannv2::migrate_ptr(ptr, gpu, torch_stream);
    // Fake a kernel on the stream
    LBANNV2_CHECK_GPU(hipStreamWaitValue32(
      torch_stream, wait_mem, wait_value, hipStreamWaitValueEq));
  }
  // GPU allocation will "FREE_REQUESTED" here, but it should NOT be
  // available for reuse

  auto ptr = alloc.allocate(size);
  CHECK(ptr.get() != orig_ptr);  // NOT REQUIRE -- need to clean up.

  // Write the new value
  *wait_mem = wait_value;

  // Ensure the "kernel" is done.
  torch_stream.synchronize();

  // Free our wait memory
  LBANNV2_CHECK_GPU(hipFree(wait_mem));
}
