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
 *  the memory between the "cpu" backend and the "cuda" backend, under
 *  certain circumstances. The semantics differ from the "to" operator
 *  in the sense that the original tensor is considered "invalid"
 *  (implicitly, of course) after the migration.
 *
 *  The primary prerequisite for migrating a tensor is that its
 *  backing memory must have been allocated using a "cuda" allocator
 *  (that is, somewhere in the allocator stack, the raw memory must
 *  come from "hipMalloc" in the case of MI300A). LBANNv2 provides a
 *  context manager that replaces the underlying CPU allocator with
 *  one that allocates "cuda" memory, essentially providing
 *  "migrateable" CPU tensors.
 *
 *  At this time, we do NOT support IPC memory buffers or P2P device
 *  memory access. Thus, tensors are only migrateable between the CPU
 *  and whichever CUDA device their allocation is tied to. In the case
 *  of CPU tensors allocated using the LBANNv2 allocator, this will be
 *  whichever CUDA device was selected at the time of its allocation.
 *
 *  If we do not have an APU, this is just a direct call to "to".
 *
 *  Upon successful migration, the input tensor is invalidated to
 *  prevent foot wounds.
 *
 *  Schema: migrate(Tensor(a!), Device) -> Tensor(a!)
 *
 *  @param[in] t The tensor to (possibly) migrate.
 *  @param[in] d The target device.
 *
 *  @returns A tensor associated with the given target device.
 */
at::Tensor migrate(at::Tensor& t, c10::Device const& d);

}// namespace lbannv2
