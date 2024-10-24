////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include <lbannv2_config.h>

#include "hooks_interface.hpp"

#include <lbannv2/utils/errors.hpp>

#if LBANNV2_HAS_GPU
#include <h2/gpu/runtime.hpp>
#endif  // LBANNV2_HAS_GPU

#define LBANNV2_NOT_IMPLEMENTED(fn)                                            \
  throw std::runtime_error("Not implemented: " fn)

namespace lbannv2
{
bool LBANNv2HooksInterface::hasPrimaryContext(
  [[maybe_unused]] c10::DeviceIndex const device_index) const
{
  // Explected indices are 0 ("cpu") and 1 ("gpu").
#if LBANNV2_HAS_GPU
  LBANNV2_ASSERT_DEBUG(device_index == 0 || device_index == 1);
  return device_index == 0 || h2::gpu::runtime_is_initialized();
#else
  LBANNV2_ASSERT_DEBUG(device_index == 0);
  return true;
#endif
}

c10::DeviceIndex LBANNv2HooksInterface::deviceCount() const
{
#if LBANNV2_HAS_GPU
  return 1 + (h2::gpu::num_gpus() > 0);
#else
  return 1;
#endif
}

void LBANNv2HooksInterface::setCurrentDevice(
  c10::DeviceIndex const device_index) const
{
#if LBANNV2_HAS_GPU
  (void) device_index;
  LBANNV2_NOT_IMPLEMENTED("LBANNv2HooksInterface::setCurrentDevice");
#else
  LBANNV2_ASSERT_ALWAYS(device_index == 0);
#endif
}

c10::DeviceIndex LBANNv2HooksInterface::getCurrentDevice() const
{
#if LBANNV2_HAS_GPU
  LBANNV2_NOT_IMPLEMENTED("LBANNv2HooksInterface::getCurrentDevice");
#else
  return 0;
#endif
}

c10::DeviceIndex
LBANNv2HooksInterface::exchangeDevice(c10::DeviceIndex const device_index) const
{
#if LBANNV2_HAS_GPU
  (void) device_index;
  LBANNV2_NOT_IMPLEMENTED("LBANNv2HooksInterface::exchangeDevice");
#else
  LBANNV2_ASSERT_ALWAYS(device_index == 0);
  return 0;
#endif
}

c10::DeviceIndex LBANNv2HooksInterface::maybeExchangeDevice(
  c10::DeviceIndex const device_index) const
{
  return exchangeDevice(device_index);
}

bool LBANNv2HooksInterface::isPinnedPtr(void const* const /*ptr*/) const
{
  return false;
}

c10::Allocator* LBANNv2HooksInterface::getPinnedMemoryAllocator() const
{
  LBANNV2_NOT_IMPLEMENTED("LBANNv2HooksInterface::getPinnedMemoryAllocator");
}

at::Device LBANNv2HooksInterface::getDeviceFromPtr(void* const) const
{
  LBANNV2_NOT_IMPLEMENTED("LBANNv2HooksInterface::getDeviceFromPtr");
}

at::Generator const&
LBANNv2HooksInterface::getDefaultGenerator(c10::DeviceIndex const) const
{
  LBANNV2_NOT_IMPLEMENTED("LBANNv2HooksInterface::getDefaultGenerator");
}

void LBANNv2HooksInterface::resizePrivateUse1Bytes(c10::Storage const&,
                                                   size_t const) const
{
  LBANNV2_NOT_IMPLEMENTED("LBANNv2HooksInterface::resizePrivateUse1Bytes");
}

}  // namespace lbannv2

lbannv2::LBANNv2HooksInterface* lbannv2::get_lbannv2_hooks()
{
  // This will leak; stateless, so probably just the vtable. I feel
  // terrible, absolutely gutted, about this, but this decision is
  // aligned with the choices made and more verbosely justified by
  // other backends included in PyTorch.
  static lbannv2::LBANNv2HooksInterface* lbannv2_hooks = nullptr;
  static std::once_flag flag;
  std::call_once(flag, []() {
    lbannv2_hooks =
      new lbannv2::LBANNv2HooksInterface(lbannv2::LBANNv2HooksArgs {});
  });
  return lbannv2_hooks;
}

// FIXME (trb): See about the registry business (C10_DEFINE_REGISTRY, etc)
