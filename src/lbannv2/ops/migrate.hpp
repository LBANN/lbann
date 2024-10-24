////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <lbannv2_config.h>

#include <ATen/Tensor.h>
#include <c10/core/Device.h>

namespace lbannv2
{

/** @brief Migrate a tensor to a new device, eliding copies when
 *         possible.
 *
 *  If we have an APU (e.g., MI300A), we are able to zero-copy migrate
 *  the memory CPU <-> GPU. The semantics differ from "to" in that the
 *  original tensor is considered "invalid" (implicitly, of course)
 *  after the migration.
 *
 *  Additionally, on APU systems on which we have taken over the
 *  native PyTorch allocators, we can support migration of native
 *  Torch backend tensors (since we own the memory).
 *
 *  The behavior of this function depends on two bits of "external"
 *  state: the presence of APUs and whether LBANNv2 controls the
 *  memory allocators for native Torch backends. If LBANNv2 controls
 *  the native backend allocators, we can support zero-copy migration
 *  c10::kCPU <-> {lbannv2::LBANNDeviceT, 0} (and c10::kCUDA <->
 *  {LBANNDeviceT, 1} when GPU support is enabled). If LBANNv2 has APU
 *  support, we can move copylessly between CPU and GPU within LBANN
 *  and outside of LBANN if we also control those allocators.
 *
 *  If we do not have an APU, this is just a direct call to "to".
 *
 *  The input tensor is invalidated to prevent foot wounds.
 *
 *  FIXME (trb): Get the op wrapper working.
 *  Schema: migrate(Tensor(a!), Device) -> Tensor(a!)
 */
at::Tensor migrate(at::Tensor& t, c10::Device const& d);

}// namespace lbannv2
