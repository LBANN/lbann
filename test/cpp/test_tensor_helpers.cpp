////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include <lbannv2/ops/empty_tensor.hpp>
#include <lbannv2/utils/tensor_helpers.hpp>

#include <ATen/EmptyTensor.h>

// A c10 header file in PyTorch has left a macro called `CHECK`
// defined. To prevent warnings, we need to clear that out. This
// should not cause problems as we don't use the PyTorch macro
// directly, and all PyTorch includes should precede this line in this
// source code.
#ifdef CHECK
#undef CHECK
#endif

#include <catch2/catch_test_macros.hpp>

TEST_CASE("alias_as_device", "[tensor][utils]")
{
  SECTION("Aliasing from LBANN to native device")
  {
    at::Tensor t = lbannv2::empty_lbann({2, 3, 4},
                                        c10::ScalarType::Float,
                                        std::nullopt,
                                        std::nullopt,
                                        false,
                                        std::nullopt);
    auto const orig_keys = t.key_set();
    auto const orig_device = t.device();

    at::Tensor cpu_alias = lbannv2::alias_as_device(
      t, c10::DeviceType::CPU, c10::DispatchKeySet {c10::DispatchKey::CPU});

    CHECK(t.is_privateuseone());
    CHECK(t.key_set() == orig_keys);
    CHECK(t.device() == orig_device);

    CHECK(cpu_alias.is_alias_of(t));
    CHECK(cpu_alias.is_cpu());

    // This is documented to change
    CHECK(t.storage().data_ptr().device().is_cpu());

    // Metadata should match
    CHECK(cpu_alias.sizes() == t.sizes());
    CHECK(cpu_alias.strides() == t.strides());
    CHECK(cpu_alias.names() == t.names());
    CHECK(cpu_alias.dtype() == t.dtype());
  }
}

TEST_CASE("alias_as_native_device", "[tensor][utils]")
{
  SECTION("Aliasing a native PyTorch tensor does nothing")
  {
    at::Tensor t = at::detail::empty_cpu({3, 2, 4},
                                         c10::ScalarType::Float,
                                         std::nullopt,
                                         std::nullopt,
                                         std::nullopt,
                                         std::nullopt);
    at::Tensor alias = lbannv2::alias_as_native_device(t);
    CHECK(alias.is_alias_of(t));
    CHECK(alias.key_set() == t.key_set());
    CHECK(alias.device() == t.device());
    CHECK(alias.dtype() == t.dtype());
    CHECK(alias.unsafeGetTensorImpl() == t.unsafeGetTensorImpl());
  }

  SECTION("Aliasing an LBANN tensor is ok")
  {
    using namespace lbannv2;
    static constexpr auto LBANNbit = c10::BackendComponent::PrivateUse1Bit;

    at::Tensor t = lbannv2::empty_lbann({2, 3, 4},
                                        c10::ScalarType::Float,
                                        std::nullopt,
                                        c10::Device {LBANNDeviceT, LBANN_CPU},
                                        false,
                                        std::nullopt);
    at::Tensor lbann_alias = lbannv2::alias_as_native_device(t);

    // Still an alias (based on storage objects)
    CHECK(lbann_alias.is_alias_of(t));
    CHECK(lbann_alias.key_set() == t.key_set().remove_backend(LBANNbit));
    CHECK(lbann_alias.sizes() == t.sizes());
    CHECK(lbann_alias.strides() == t.strides());
    CHECK(lbann_alias.dtype() == t.dtype());
    CHECK(lbann_alias.device() != t.device());
    CHECK(lbann_alias.device().is_cpu());
    CHECK(lbann_alias.unsafeGetTensorImpl()->data()
          == t.unsafeGetTensorImpl()->data());
    CHECK(lbann_alias.unsafeGetTensorImpl()->storage_offset()
          == t.unsafeGetTensorImpl()->storage_offset());
  }
}
