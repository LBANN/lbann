////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include <lbannv2/backend/device_guard.hpp>
#include <lbannv2/memory/toplevel_allocator.hpp>
#include <lbannv2/utils/device_helpers.hpp>

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <c10/core/Device.h>

#include <pybind11/pybind11.h>

// Device guard
namespace at::detail
{

// NB: The first macro arg will be appended to "c10::DeviceType::", so
// we cannot use "LBANNDeviceT" here.
C10_REGISTER_GUARD_IMPL(PrivateUse1, lbannv2::DeviceGuardImpl);

}  // namespace at::detail

// Generic backend allocator
REGISTER_ALLOCATOR(lbannv2::LBANNDeviceT, &::lbannv2::get_allocator());

// FIXME (trb): It's not clear to me that this needs to be here. I
// maybe want to just call this from init_lbann() -- a line of C++ is
// less offensive than the same line in Python.
namespace
{

// FIXME (trb): IMPLEMENT FOR REAL
class PrivateGeneratorImpl : public at::CPUGeneratorImpl
{
public:
  // Constructors
  PrivateGeneratorImpl(c10::DeviceIndex device_index)
  {
    device_ = c10::Device(lbannv2::LBANNDeviceT, device_index);
    key_set_ = c10::DispatchKeySet(lbannv2::LBANNDispKey);
  }
  ~PrivateGeneratorImpl() override = default;
};

// this is used to register generator
at::Generator make_generator_privateuse1(c10::DeviceIndex device_index)
{
  return at::make_generator<PrivateGeneratorImpl>(device_index);
}

void register_generator()
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  REGISTER_GENERATOR_PRIVATEUSE1(make_generator_privateuse1)
#pragma GCC diagnostic pop
}

} // namespace

namespace _lbannv2
{
void add_pytorch_support(pybind11::module_& m)
{
  m.def("register_generator",
        &register_generator,
        "Register LBANNv2 generator with ATen.");
}
} // namespace _lbannv2
