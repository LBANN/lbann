////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include <lbannv2_config.h>

#include <lbannv2/ops/empty_tensor.hpp>
#include <lbannv2/utils/device_helpers.hpp>

#include <ATen/Tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

namespace
{
// This factory function can throw, and we cannot wrap an assignment
// in Catch's `REQUIRE_NOTHROW`/`REQUIRE_THROWS*` macros. We use this
// simple wrapper to facilitate things. eglot+clangd is able to still
// forward the inlay hints from `empty_lbann` out to the
// `make_empty_tensor` signature, so that's cool.
template <typename... Args>
void make_empty_tensor(at::Tensor& t, Args&&... args)
{
  t = lbannv2::empty_lbann(std::forward<Args>(args)...);
}
}  // namespace

TEST_CASE("empty_lbann", "[ops][empty]")
{
  at::Tensor t;
  c10::Device lbann_cpu {lbannv2::LBANNDeviceT, 0},
    lbann_gpu {lbannv2::LBANNDeviceT, 1};
  SECTION("Zero-size tensor is ok")
  {
#if LBANNV2_HAS_GPU
    auto lbann_device = GENERATE_COPY(values({lbann_cpu, lbann_gpu}));
#else
    auto lbann_device = lbann_cpu;
#endif

    REQUIRE_NOTHROW(make_empty_tensor(t,
                                      c10::IntArrayRef {0},
                                      c10::ScalarType::Float,
                                      std::nullopt,
                                      lbann_device,
                                      false,
                                      std::nullopt));
    REQUIRE(t.dim() == 1);
    REQUIRE(t.sizes() == c10::IntArrayRef {0});
    REQUIRE(t.strides() == c10::IntArrayRef {1});
    REQUIRE(t.is_privateuseone());
    REQUIRE(t.dtype().toScalarType() == c10::ScalarType::Float);
    REQUIRE(t.key_set().has(lbannv2::LBANNDispKey));
    REQUIRE_FALSE(t.is_pinned());
  }

  SECTION("Nonzero tensor is ok")
  {
#if LBANNV2_HAS_GPU
    auto lbann_device = GENERATE_COPY(values({lbann_cpu, lbann_gpu}));
#else
    auto lbann_device = lbann_cpu;
#endif
    REQUIRE_NOTHROW(make_empty_tensor(t,
                                      c10::IntArrayRef {3, 4},
                                      c10::ScalarType::Float,
                                      std::nullopt,
                                      lbann_device,
                                      false,
                                      std::nullopt));
    REQUIRE(t.dim() == 2);
    REQUIRE(t.sizes() == c10::IntArrayRef {3, 4});
    REQUIRE(t.strides() == c10::IntArrayRef {4, 1});
    REQUIRE(t.is_privateuseone());
    REQUIRE(t.dtype().toScalarType() == c10::ScalarType::Float);
    REQUIRE(t.key_set().has(lbannv2::LBANNDispKey));
    REQUIRE_FALSE(t.is_pinned());

    REQUIRE_NOTHROW(make_empty_tensor(t,
                                      c10::IntArrayRef {2, 3, 4, 5},
                                      c10::ScalarType::Float,
                                      std::nullopt,
                                      lbann_device,
                                      false,
                                      std::nullopt));
    REQUIRE(t.dim() == 4);
    REQUIRE(t.sizes() == c10::IntArrayRef {2, 3, 4, 5});
    REQUIRE(t.strides() == c10::IntArrayRef {60, 20, 5, 1});
    REQUIRE(t.is_privateuseone());
    REQUIRE(t.dtype().toScalarType() == c10::ScalarType::Float);
    REQUIRE(t.key_set().has(lbannv2::LBANNDispKey));
    REQUIRE_FALSE(t.is_pinned());
  }

  SECTION("Non-LBANN devices throw")
  {
    REQUIRE_THROWS_WITH(
      make_empty_tensor(t,
                        c10::IntArrayRef {3, 4},
                        c10::ScalarType::Float,
                        std::nullopt,
                        c10::DeviceType::CPU,
                        false,
                        std::nullopt),
      "LBANN should only be constructing tensors on \"PrivateUse1\" backend");
  }
}

namespace
{

template <typename... Args>
void make_empty_strided_tensor(at::Tensor& t, Args&&... args)
{
  t = lbannv2::empty_strided_lbann(std::forward<Args>(args)...);
}

}  // namespace

TEST_CASE("empty_strided_lbann", "[ops][empty]")
{
  at::Tensor t;
  c10::Device lbann_cpu {lbannv2::LBANNDeviceT, 0},
    lbann_gpu {lbannv2::LBANNDeviceT, 1};

  SECTION("Zero-size tensor is ok")
  {
#if LBANNV2_HAS_GPU
    auto lbann_device = GENERATE_COPY(values({lbann_cpu, lbann_gpu}));
#else
    auto lbann_device = lbann_cpu;
#endif
    REQUIRE_NOTHROW(make_empty_strided_tensor(t,
                                              c10::IntArrayRef {0},
                                              c10::IntArrayRef {1},
                                              c10::ScalarType::Float,
                                              std::nullopt,
                                              lbann_device,
                                              false));
    REQUIRE(t.dim() == 1);
    REQUIRE(t.sizes() == c10::IntArrayRef {0});
    REQUIRE(t.strides() == c10::IntArrayRef {1});
    REQUIRE(t.is_privateuseone());
    REQUIRE(t.dtype().toScalarType() == c10::ScalarType::Float);
    REQUIRE(t.key_set().has(lbannv2::LBANNDispKey));
    REQUIRE_FALSE(t.is_pinned());
  }

  SECTION("Nonzero tensor is ok")
  {
#if LBANNV2_HAS_GPU
    auto lbann_device = GENERATE_COPY(values({lbann_cpu, lbann_gpu}));
#else
    auto lbann_device = lbann_cpu;
#endif
    REQUIRE_NOTHROW(make_empty_strided_tensor(t,
                                              c10::IntArrayRef {3, 4},
                                              c10::IntArrayRef {8, 2},
                                              c10::ScalarType::Float,
                                              std::nullopt,
                                              lbann_device,
                                              false));
    REQUIRE(t.dim() == 2);
    REQUIRE(t.sizes() == c10::IntArrayRef {3, 4});
    REQUIRE(t.strides() == c10::IntArrayRef {8, 2});
    REQUIRE(t.is_privateuseone());
    REQUIRE(t.dtype().toScalarType() == c10::ScalarType::Float);
    REQUIRE(t.key_set().has(lbannv2::LBANNDispKey));
    REQUIRE_FALSE(t.is_pinned());

    REQUIRE_NOTHROW(make_empty_strided_tensor(t,
                                              c10::IntArrayRef {2, 3, 4, 5},
                                              c10::IntArrayRef {120, 40, 10, 2},
                                              c10::ScalarType::Float,
                                              std::nullopt,
                                              std::nullopt,
                                              false));
    REQUIRE(t.dim() == 4);
    REQUIRE(t.sizes() == c10::IntArrayRef {2, 3, 4, 5});
    REQUIRE(t.strides() == c10::IntArrayRef {120, 40, 10, 2});
    REQUIRE(t.is_privateuseone());
    REQUIRE(t.dtype().toScalarType() == c10::ScalarType::Float);
    REQUIRE(t.key_set().has(lbannv2::LBANNDispKey));
    REQUIRE_FALSE(t.is_pinned());
  }

  SECTION("Non-LBANN devices throw")
  {
    REQUIRE_THROWS_WITH(
      make_empty_strided_tensor(t,
                                c10::IntArrayRef {3, 4},
                                c10::IntArrayRef {8, 2},
                                c10::ScalarType::Float,
                                std::nullopt,
                                c10::DeviceType::CPU,
                                false),
      "LBANN should only be constructing tensors on \"PrivateUse1\" backend");
  }
}
