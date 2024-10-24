////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include "empty_tensor.hpp"

#include "lbannv2/backend/device_guard.hpp"
#include <lbannv2/backend/library_state.hpp>
#include <lbannv2/memory/toplevel_allocator.hpp>
#include <lbannv2/utils/logging.hpp>
#include <lbannv2/utils/tensor_helpers.hpp>

#include <ATen/EmptyTensor.h>

namespace
{
c10::Device device_or_current(std::optional<c10::Device> device_opt)
{
  if (device_opt)
  {
    auto const device = *device_opt;
    LBANNV2_ASSERT(
      lbannv2::is_lbann(device),
      std::runtime_error,
      "LBANN should only be constructing tensors on \"PrivateUse1\" backend");
    return device;
  }
  auto const dev = lbannv2::state::current_device_lbann();
  LBANNV2_DEBUG("empty_lbann: no device provided. using d={}", dev.str());
  return dev;
}

c10::ScalarType datatype_or_current(std::optional<c10::ScalarType> dtype_opt)
{
  return dtype_opt.value_or(lbannv2::state::current_dtype());
}

// FIXME (trb): Maybe move to types.hpp or utils/<something>?
c10::DispatchKey get_dispatch_key_for(c10::Device const& d)
{
  switch (d.type())
  {
  case c10::DeviceType::CPU: return c10::DispatchKey::CPU;
  case c10::DeviceType::CUDA: return c10::DispatchKey::CUDA;
  case c10::DeviceType::HIP: return c10::DispatchKey::HIP;
  case lbannv2::LBANNDeviceT:
    return get_dispatch_key_for(lbannv2::to_native(d));
  default:
    throw std::runtime_error("Unknown device type: "
                             + c10::DeviceTypeName(d.type()));
  }
}

c10::DispatchKeySet get_dispatch_keyset(c10::Device const& d)
{
  return c10::DispatchKeySet {lbannv2::LBANNDispKey,
                              get_dispatch_key_for(lbannv2::to_native(d))};
}

}  // namespace

at::TensorBase lbannv2::empty_lbann(c10::IntArrayRef size,
                                    c10::TensorOptions const& options)
{
  return empty_lbann(size,
                     c10::optTypeMetaToScalarType(options.dtype_opt()),
                     options.layout_opt(),
                     options.device_opt(),
                     options.pinned_memory_opt(),
                     options.memory_format_opt());
}

// I'm wondering if we should, in fact, allow non-LBANN devices for
// `device_opt`. The behavior would then be: nullopt or lbann ->
// consult lbannv2 current library state, non-null non-lbann ->
// allocate lbann memory on that device if known or throw.
at::TensorBase
lbannv2::empty_lbann(c10::IntArrayRef size,
                     std::optional<c10::ScalarType> dtype_opt,
                     std::optional<c10::Layout> layout_opt,
                     std::optional<c10::Device> device_opt,
                     std::optional<bool> pin_memory_opt,
                     std::optional<c10::MemoryFormat> memory_format_opt)
{
  if (layout_opt.has_value())
    LBANNV2_ASSERT(*layout_opt == c10::Layout::Strided,
                   std::runtime_error,
                   "LBANN only supports \"Strided\" layout");

  auto const device = device_or_current(device_opt);
  auto const dtype = datatype_or_current(dtype_opt);
  auto const keyset = get_dispatch_keyset(device);
  auto const pinned = pin_memory_opt.value_or(false);

  LBANNDeviceGuard device_guard(device);

  LBANNV2_ASSERT(is_supported(dtype),
                 std::runtime_error,
                 std::string {"Unsupported LBANN datatype: "}
                   + c10::toString(dtype));

  LBANNV2_TRACE("empty_lbann(size={}, device={}, dtype={}, keys={})",
                to_str(size),
                device.str(),
                c10::toString(dtype),
                c10::toString(keyset));

  return at::detail::empty_generic(
    size,
    &get_allocator(device, pinned),
    //&::lbannv2::get_managed_allocator(device, pinned),
    keyset,
    dtype,
    memory_format_opt);
}

at::TensorBase lbannv2::empty_strided_lbann(c10::IntArrayRef size,
                                            c10::IntArrayRef stride,
                                            c10::TensorOptions const& options)
{
  return empty_strided_lbann(size,
                             stride,
                             c10::optTypeMetaToScalarType(options.dtype_opt()),
                             options.layout_opt(),
                             options.device_opt(),
                             options.pinned_memory_opt());
}

at::TensorBase
lbannv2::empty_strided_lbann(c10::IntArrayRef size,
                             c10::IntArrayRef stride,
                             std::optional<c10::ScalarType> dtype_opt,
                             std::optional<c10::Layout> layout_opt,
                             std::optional<c10::Device> device_opt,
                             std::optional<bool> pin_memory_opt)
{
  if (layout_opt.has_value())
    LBANNV2_ASSERT(*layout_opt == c10::Layout::Strided,
                   std::runtime_error,
                   "LBANN only supports \"Strided\" layout");

  auto const device = device_or_current(device_opt);
  auto const dtype = datatype_or_current(dtype_opt);
  auto const pinned = pin_memory_opt.value_or(false);
  auto const keyset = get_dispatch_keyset(device);

  LBANNDeviceGuard device_guard(device);

  LBANNV2_TRACE(
    "empty_strided_lbann(size={}, stride={}, device={}, dtype={}, keys={})",
    to_str(size),
    to_str(stride),
    device.str(),
    c10::toString(dtype),
    c10::toString(keyset));

  return at::detail::empty_strided_generic(
    size, stride, &get_allocator(device, pinned), keyset, dtype);
}
