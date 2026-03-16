////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include <lbannv2_config.h>

#include <lbannv2/memory/memory_utils.hpp>
#include <lbannv2/memory/registry.hpp>
#include <lbannv2/ops/migrate.hpp>
#include <lbannv2/utils/logging.hpp>

#if LBANNV2_HAS_GPU
#include <lbannv2/utils/gpu_utils.hpp>
#endif

#include <lbannv2/memory/allocator.hpp>

#include <ATen/ops/to_native.h>
#include <c10/core/Device.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/extension.h>
#include <torch/library.h>

namespace
{

// Migrate
at::Tensor py_migrate(at::Tensor& t, at::Device const& d)
{
  return lbannv2::migrate(t, d);
}

bool py_supports_migrate() noexcept
{
#if LBANNV2_WITH_MI300A
  return true;
#elif LBANNV2_HAS_GPU
  return lbannv2::gpu::is_integrated();
#else
  return false;
#endif
}

void py_use_mi300a_host_allocator()
{
  lbannv2::use_mi300a_cpu_allocator();
}

void py_use_torch_host_allocator()
{
  lbannv2::use_torch_cpu_allocator();
}

bool py_using_lbannv2_memory(torch::Tensor const& t)
{
  return lbannv2::pointer_registry().known(t.const_data_ptr());
}

}  // namespace

namespace _lbannv2
{

void add_memory_funcs(pybind11::module_& m)
{
  // Pointer migration
  m.def("supports_migrate",
        &py_supports_migrate,
        "Determine whether device migration is supported");

  m.def("migrate",
        &py_migrate,
        "Try to migrate an LBANNv2-owned pointer to a new device.");

  m.def("use_mi300a_host_allocator",
        &py_use_mi300a_host_allocator,
        "Use the LBANNv2 MI300A allocator for CPU allocations");

  m.def("use_pytorch_host_allocator",
        &py_use_torch_host_allocator,
        "Use the default pytorch CPU allocator for CPU allocations");

  m.def(
    "using_lbannv2_memory",
    &py_using_lbannv2_memory,
    "Determine whether LBANNv2 allocated the memory backing a given tensor");
}

}  // namespace _lbannv2
