////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include <lbannv2_config.h>

#include <lbannv2/backend/library_state.hpp>
#include <lbannv2/memory/memory_utils.hpp>
#include <lbannv2/memory/registry.hpp>
#include <lbannv2/memory/toplevel_allocator.hpp>
#include <lbannv2/utils/device_helpers.hpp>
#include <lbannv2/utils/logging.hpp>

#if LBANNV2_WITH_MI300A || LBANNV2_UNKNOWN_MI300A
#include <lbannv2/memory/mi300a_allocator.hpp>
#include <lbannv2/ops/migrate.hpp>
#endif

#include <unordered_map>

#include <c10/core/Device.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

// FIXME (trb): WHERE TO PUT THIS???
namespace
{
// These must persist -- following the letter of the c10 docs, the
// alloc passed to SetAllocator must have static storage duration.
// (However, I don't think this is actually all that essential in this
// case because the deleter that's registered with created DataPtrs is
// just a reference to the underlying allocator, which has static
// storage duration independent of this nonsense.)
std::unordered_map<c10::DeviceType, lbannv2::AllocatorWrapper> _wrapped_allocs;
std::unordered_map<c10::DeviceType, c10::Allocator*> _alloc_stash;
void use_lbannv2_allocator_for(c10::Device const& device)
{
  auto const device_type = device.type();
  // We already own PU1 allocs; otherwise, check if we already own the alloc
  if (device_type == c10::kPrivateUse1 || _alloc_stash.count(device_type))
    return;

  LBANNV2_TRACE("Using LBANNv2 allocator for device {}", device.str());

  _alloc_stash[device_type] = c10::GetAllocator(device_type);

  auto [it, _] = _wrapped_allocs.try_emplace(
    device_type,
    lbannv2::get_allocator(lbannv2::to_lbann(device), false),
    device);
  c10::SetAllocator(device_type, &(it->second));
}

void use_lbannv2_allocators()
{
  use_lbannv2_allocator_for(c10::kCPU);
  if (lbannv2::state::has_gpu())
    use_lbannv2_allocator_for({c10::kCUDA, lbannv2::state::gpu_idx()});
}

void restore_default_allocator_for(c10::Device const& device)
{
  auto it = _alloc_stash.find(device.type());
  if (it != _alloc_stash.end())
  {
    LBANNV2_TRACE("Restoring default allocator for device {}", device.str());
    c10::SetAllocator(device.type(), it->second);
    _alloc_stash.erase(it);
  }
}

void restore_default_allocators()
{
  restore_default_allocator_for(c10::kCPU);
  if (lbannv2::state::has_gpu())
    restore_default_allocator_for({c10::kCUDA, lbannv2::state::gpu_idx()});
}


#if LBANNV2_WITH_MI300A || LBANNV2_UNKNOWN_MI300A
// Migrate
at::Tensor py_migrate(at::Tensor& t, at::Device const& d)
{
  return lbannv2::migrate(t, d);
}
#endif

bool py_lbannv2_knows_ptr(at::Tensor const& t)
{
  return lbannv2::pointer_registry().known(t.const_data_ptr());
}

}  // namespace

namespace _lbannv2
{

void add_memory_funcs(pybind11::module_& m)
{
  // Memory knowledge
  m.def("using_lbannv2_memory",
        &py_lbannv2_knows_ptr,
        "Determine if the tensor is backed by LBANNv2 memory");

#if LBANNV2_WITH_MI300A || LBANNV2_UNKNOWN_MI300A
  // Pointer migration
  m.def("migrate",
        &py_migrate,
        "Migrate an LBANNv2-owned pointer to a new device.");
#endif

  // Allocator management
  m.def("use_lbannv2_allocator_for",
        &use_lbannv2_allocator_for,
        "Replace a Torch/C10 allocator with the LBANNv2 allocator");
  m.def("use_lbannv2_allocators",
        &use_lbannv2_allocators,
        "Replace all Torch/C10 allocators with their LBANNv2 counterparts");
  m.def("restore_default_allocator_for",
        &restore_default_allocator_for,
        "Restore the Torch/C10 for the given device.");
  m.def("restore_default_allocators",
        &restore_default_allocators,
        "Restore all Torch/C10 allocators");
}

}  // namespace _lbannv2
