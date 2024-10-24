////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <lbannv2_export.h>

#include <ATen/detail/PrivateUse1HooksInterface.h>

namespace lbannv2
{

// FIXME (trb): idk if we need a real (sub)class for HooksArgs --
// c10::PrivateUse1HooksArgs is not abstract (it's an empty, trivial
// struct). Since I don't seem to actually need them, I'm going to
// ignore them as long as hooks registration goes through with the
// default class.
using LBANNv2HooksArgs = at::PrivateUse1HooksArgs;

// It looks like a fair number of the examples included with PyTorch
// do some indirection gymnastics (<Backend>HooksInterface just
// throws, <Backend>Hooks implements, and the latter is constructed
// iff the backend is enabled). I'm NOT doing that because LBANNv2 is
// always available when building and using LBANNv2. Crazy right??
struct LBANNV2_EXPORT LBANNv2HooksInterface final
  : public at::PrivateUse1HooksInterface
{
  LBANNv2HooksInterface(LBANNv2HooksArgs) {}
  virtual ~LBANNv2HooksInterface() = default;

  /** @name AcceleratorHooksInterface interface */
  ///@{

  bool hasPrimaryContext(c10::DeviceIndex) const final;

  c10::DeviceIndex deviceCount() const final;
  void setCurrentDevice(c10::DeviceIndex) const final;
  c10::DeviceIndex getCurrentDevice() const final;
  c10::DeviceIndex exchangeDevice(c10::DeviceIndex) const final;
  c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex) const final;

  bool isPinnedPtr(void const*) const final;
  c10::Allocator* getPinnedMemoryAllocator() const final;
  at::Device getDeviceFromPtr(void*) const final;

  ///@}
  /** @name Specific PrivateUse1HooksInterface interface */
  ///@{
  at::Generator const& getDefaultGenerator(c10::DeviceIndex) const final;
  void resizePrivateUse1Bytes(c10::Storage const&, size_t) const final;
  ///@}
};  // struct LBANNv2HooksInterface

lbannv2::LBANNv2HooksInterface* get_lbannv2_hooks();

}  // namespace lbannv2
