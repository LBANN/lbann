////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include <lbannv2_config.h>

#include <lbannv2/backend/hooks_interface.hpp>
#include <lbannv2/backend/library_state.hpp>
#include <lbannv2/utils/device_helpers.hpp>
#include <lbannv2/utils/logging.hpp>

#if LBANNV2_HAS_GPU
#include <h2/gpu/runtime.hpp>
#endif

#include <cstdlib>
#include <stdexcept>

#include <c10/core/Device.h>
#include <pybind11/pybind11.h>

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

  if (c10::is_privateuse1_backend_registered())
    throw std::runtime_error("Cannot register LBANNv2 with PyTorch. "
                             "PrivateUse1 backend is already registered!");

  c10::register_privateuse1_backend("lbann");
  at::RegisterPrivateUse1HooksInterface(lbannv2::get_lbannv2_hooks());

  _lbannv2_initialized = true;
}

void init_lbannv2_gpu()
{
  if (!_lbannv2_initialized)
    init_lbannv2();

#if LBANNV2_HAS_GPU
  if (_lbannv2_gpu_initialized)
    return;

  h2::gpu::init_runtime();
  LBANNV2_ASSERT(lbannv2::state::gpu_idx() == h2::gpu::current_gpu(),
                 std::runtime_error,
                 "GPU device id mismatch");
  lbannv2::state::set_device(
    c10::Device {lbannv2::LBANNDeviceT, lbannv2::LBANN_GPU});

  _lbannv2_gpu_initialized = true;
#endif
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
void add_pytorch_support(pybind11::module_& m);
}  // namespace _lbannv2

PYBIND11_MODULE(_lbannv2, m)
{
  m.def("init_lbannv2", &init_lbannv2, "Initialize state for LBANNv2");
  m.def("init_lbannv2_gpu", &init_lbannv2_gpu, "Initialize state for LBANNv2");
  m.def("is_lbannv2_initialized",
        &is_lbannv2_initialized,
        "Query initialization state for LBANNv2");
  m.def("is_lbannv2_gpu_initialized",
        &is_lbannv2_gpu_initialized,
        "Query initialization state for LBANNv2 GPU support.");
  m.def("is_lbannv2_gpu_available",
        &is_lbannv2_gpu_available,
        "Query whether LBANNv2 has GPU support.");

  _lbannv2::add_memory_funcs(m);
  _lbannv2::add_pytorch_support(m);
}
