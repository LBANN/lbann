////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include <lbannv2/memory/h2_allocator_wrappers.hpp>

namespace lbannv2
{

template <h2::Device D>
H2AllocatorWrapper<D>& H2AllocatorWrapper<D>::instance()
{
  static H2AllocatorWrapper<D> allocator;
  return allocator;
}

}  // namespace lbannv2
