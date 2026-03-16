////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include <lbannv2_config.h>

#include <lbannv2/utils/gpu_utils.hpp>
#include <lbannv2/utils/logging.hpp>

#include <c10/core/Device.h>
#include <pybind11/pybind11.h>

#include <cstdlib>
#include <iostream>

#include <sys/types.h>
#include <unistd.h>

namespace
{
bool _lbannv2_initialized = false;
bool _lbannv2_gpu_initialized = false;

void init_lbannv2()
{
  if (_lbannv2_initialized)
    return;

  if (std::getenv("LBANNV2_HANG_FOR_DEBUG"))
  {
    // Raw vs spdlog here because I want to force the flush.
    std::cout << "LBANNV2 WAITING ON PID " << getpid() << std::endl;
    int volatile wait = 1;
    while (wait) {}
  }

#if LBANNV2_HAS_GPU
  if (!_lbannv2_gpu_initialized)
  {
    // There's nothing to do here since getting rid of H2.
    _lbannv2_gpu_initialized = true;
  }
#endif

  _lbannv2_initialized = true;
}

bool is_lbannv2_initialized() noexcept
{
  return _lbannv2_initialized;
}

bool is_lbannv2_gpu_initialized() noexcept
{
  return _lbannv2_gpu_initialized;
}

bool is_lbannv2_gpu_available() noexcept
{
  return LBANNV2_HAS_GPU;
}

}  // namespace

namespace _lbannv2
{
void add_memory_funcs(pybind11::module_& m);
}  // namespace _lbannv2

PYBIND11_MODULE(_lbannv2, m)
{
  m.def("init_lbannv2", &init_lbannv2, "Initialize state for LBANNv2");
  m.def("is_lbannv2_initialized",
        &is_lbannv2_initialized,
        "Query initialization state for LBANNv2");
  m.def("is_lbannv2_gpu_initialized",
        &is_lbannv2_gpu_initialized,
        "Query initialization state for LBANNv2 GPU support.");
  m.def("is_lbannv2_gpu_available",
        &is_lbannv2_gpu_available,
        "Query whether LBANNv2 has GPU support.");
  m.def("set_log_level",
        &lbannv2::set_log_level,
        "Set the output level for LBANNv2 logging.");

  _lbannv2::add_memory_funcs(m);
}
