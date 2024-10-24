////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include <lbannv2/backend/library_state.hpp>
#include <lbannv2/utils/device_helpers.hpp>

#if LBANNV2_HAS_GPU
#include <h2/gpu/runtime.hpp>
#endif

// A c10 header file in PyTorch has left a macro called `CHECK`
// defined. To prevent warnings, we need to clear that out. This
// should not cause problems as we don't use the PyTorch macro
// directly, and all PyTorch includes should precede this line in this
// source code.
#ifdef CHECK
#undef CHECK
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using namespace lbannv2;

TEST_CASE("is_lbann", "[device][utils]")
{
  REQUIRE(is_lbann(c10::Device {LBANNDeviceT}));
  REQUIRE(is_lbann(c10::Device {LBANNDeviceT, 0}));
  REQUIRE(is_lbann(c10::Device {LBANNDeviceT, 1}));
  REQUIRE_FALSE(is_lbann(c10::Device {c10::DeviceType::CPU}));
  REQUIRE_FALSE(is_lbann(c10::Device {c10::DeviceType::CUDA}));
  REQUIRE_FALSE(is_lbann(c10::Device {c10::DeviceType::MPS}));
}

TEST_CASE("to_native", "[device][utils]")
{
  REQUIRE(to_native(c10::Device {LBANNDeviceT})
          == state::current_device_native());
  REQUIRE(to_native(c10::Device {LBANNDeviceT, LBANN_CPU})
          == c10::Device {c10::kCPU});
  REQUIRE_THROWS_WITH(to_native(c10::Device {LBANNDeviceT, NumLBANNDevices}),
                      Catch::Matchers::StartsWith("Invalid device index."));

#if LBANNV2_HAS_GPU
  REQUIRE(to_native(c10::Device {LBANNDeviceT, LBANN_GPU})
          == c10::Device {LBANN_GPU_TYPE, state::gpu_idx()});
#endif
}

TEST_CASE("to_lbann", "[device][utils]")
{
  REQUIRE(to_lbann(c10::Device {c10::kCPU}) == c10::Device {LBANNDeviceT, 0});
  REQUIRE(to_lbann(c10::Device {c10::kCPU, 0})
          == c10::Device {LBANNDeviceT, 0});

  REQUIRE_THROWS_WITH(to_lbann(c10::Device {c10::kMPS}),
                      "Device type not handled by LBANN");
}

#if LBANNV2_HAS_GPU

TEST_CASE("to_lbann (GPU)", "[device][utils]")
{
  // LBANN accepts "CUDA" device in ROCm builds, so this block does
  // NOT need guards
  REQUIRE(to_lbann(c10::Device {c10::kCUDA}) == c10::Device {LBANNDeviceT, 1});
  REQUIRE(to_lbann(c10::Device {c10::kCUDA, state::gpu_idx()})
          == c10::Device {LBANNDeviceT, 1});

#if LBANNV2_HAS_ROCM
  REQUIRE(to_lbann(c10::Device {c10::kHIP}) == c10::Device {LBANNDeviceT, 1});
  REQUIRE(to_lbann(c10::Device {c10::kHIP, state::gpu_idx()})
          == c10::Device {LBANNDeviceT, 1});

  REQUIRE_THROWS_WITH(
    to_lbann(c10::Device {c10::kHIP,
                          static_cast<c10::DeviceIndex>(state::gpu_idx() + 1)}),
    Catch::Matchers::StartsWith("Invalid GPU index"));
#endif

  REQUIRE_THROWS_WITH(
    to_lbann(c10::Device {c10::kCUDA,
                          static_cast<c10::DeviceIndex>(state::gpu_idx() + 1)}),
    Catch::Matchers::StartsWith("Invalid GPU index"));

#if LBANNV2_HAS_CUDA
  // HOWEVER, LBANN does NOT accept HIP device in CUDA builds. So check
  // these throw.
  REQUIRE_THROWS_WITH(to_lbann(c10::Device {c10::kHIP}),
                      "Device type not handled by LBANN");
  REQUIRE_THROWS_WITH(to_lbann(c10::Device {c10::kHIP, state::gpu_idx()}),
                      "Device type not handled by LBANN");
#endif
}

#endif
