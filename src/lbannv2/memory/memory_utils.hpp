////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <lbannv2_config.h>

#include <lbannv2/memory/allocator.hpp>

namespace lbannv2
{

/** @class AllocatorWrapper
 *  @brief Wrap an allocator with a different device.
 *
 *  This wraps a c10::Allocator instance. Allocations from that
 *  allocator are intercepted and the DataPtr is updated to have the
 *  specified Device.
 *
 *  The primary intention is to wrap LBANN allocators as "native
 *  device" allocators, though it could be used the other way, too.
 *  However, there is no pointer registration in this class -- LBANNv2
 *  allocators handle this internally, so including that here would
 *  "double register" pointers. This could be cleaned up a bit down
 *  the road.
 */
class AllocatorWrapper : public c10::Allocator
{
public:
  /** @brief Constructor
   *
   *  @param[in] alloc The allocator to wrap.
   *  @param[in] device The device to use for DataPtrs produced by
   *                    this allocator.
   */
  AllocatorWrapper(c10::Allocator& alloc, c10::Device device)
    : m_alloc {alloc}, m_device {std::move(device)}
  {}
  ~AllocatorWrapper() = default;

  c10::DataPtr allocate(size_t n) final
  {
    auto dptr = m_alloc.allocate(n);
    dptr.unsafe_set_device(m_device);
    // NOTE (trb): We could replace the deleter fn to be
    // this->raw_deleter, but since this->raw_deleter() just calls
    // that->raw_deleter(), what would be the point?? This story
    // changes if we start tracking memory allocations in the registry
    // through this class.
    return dptr;
  }

  c10::DeleterFnPtr raw_deleter() const noexcept final
  {
    return m_alloc.raw_deleter();
  }

  void copy_data(void* dst, void const* src, size_t n) const final
  {
    m_alloc.copy_data(dst, src, n);
  }

private:
  c10::Allocator& m_alloc;
  c10::Device m_device;
};  // class AllocatorWrapper

}  // namespace lbannv2
