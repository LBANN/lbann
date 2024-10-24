////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <lbannv2_config.h>

#include <lbannv2/memory/allocator.hpp>

#include <map>
#include <mutex>
#include <stdexcept>

#include <c10/core/DeviceType.h>

namespace lbannv2
{

struct LBANNV2_EXPORT UnknownAddress : std::runtime_error
{
  UnknownAddress() : std::runtime_error {"Unknown address"} {}
};

// We should consider the issue of registering nullptr or equivalent
// zero-size allocations. Note that if ISO C++ is the only source of
// memory, this should be an error. But I'm not sure how all of the
// allocators we encounter might handle a zero-size allocation (e.g.,
// cudaMalloc and friends). ISO C++, however, requires zero-size
// allocations to still return unique, non-null pointers (section
// 6.7.5.5.2, paragraph 2).

/** @class PointerRegistry
 *  @brief Tracks known memory regions
 */
class LBANNV2_EXPORT PointerRegistry
{
public:
  /** @brief Register an allocation.
   *
   *  @param[in] ptr The beginning of the allocated range.
   *  @param[in] size The size in bytes of the allocated range.
   *  @param[in] allocator The allocator responsible for deleting the range.
   */
  void add(void* ptr, size_t size, Allocator* allocator);

  /** @brief Deregister an allocation.
   *
   *  The pointer passed must match a pointer registered with add().
   *
   *  @param[in] ptr The (context) pointer to deregister.
   */
  void remove(void* ptr);

  /** @brief Query whether this address is part of a registered
   *         allocation.
   *
   *  Returns @c true for any address that is included in a registered
   *  allocation, that is, in the range [ptr, ptr + size) for any
   *  (ptr, size) passed to add().
   *
   *  @param[in] ptr The pointer in question.
   */
  bool known(void const* ptr) const noexcept;

  /** @brief Get the allocator used to allocate this pointer.
   *
   *  @param[in] ptr The pointer whose allocator is needed.
   *
   *  @throws UnknownAddress if the pointer is not part of a
   *          registered allocation.
   */
  Allocator* get_allocator(void const* ptr) const;

  /** @brief Reset the allocator associated with a pointer.
   *
   *  In cases of MI300A pointer migration, this allows us to keep our
   *  internal bookkeeping consistent. It should not be used outside
   *  of this context.
   */
  void unsafe_reset_allocator(void const* ptr, Allocator* new_alloc);
  // FIXME (trb): An alternative would be to make this similar to
  // "compare and swap" semantics (i.e., having to provide what the
  // user thinks the current allocator is); see also, replacing a
  // deleter on a DataPtr. My concern is this will never be called
  // "properly" but rather just with a dummy
  // "registry.get_allocator(ptr)" in that argument, so what would the
  // point really be?

  /** @brief Get the context of the given pointer.
   *
   *  The context is the address returned by the raw allocator when
   *  the allocation is requested. It is the pointer that must be
   *  passed to @c delete.
   *
   *  @param[in] ptr The pointer whose context is needed.
   *
   *  @throws UnknownAddress if the pointer is not part of a
   *          registered allocation.
   */
  void* get_context(void const* ptr) const;

  /** @brief Get the current number of registered ranges */
  size_t num_registered() const noexcept
  {
    std::lock_guard<std::mutex> lock(m_registry_mtx);
    return m_registry.size();
  }

  /** @brief Get the current number of registered bytes */
  size_t bytes_registered() const noexcept;

  /** @brief Get the number of bytes associated with the given
   *         pointer.
   *
   *  Unregistered pointers return 0. Since zero-sized ranges are
   *  allowed in the registry, this function cannot serve as a proxy
   *  for known().
   *
   *  @param[in] ptr Any valid address.
   *
   *  @returns The number of bytes in an allocation associated with
   *           the pointer.
   */
  size_t bytes_registered(void const*) const noexcept;

public:
  using KeyT = std::pair<void*, void*>;
  /** @class RangeLessAndDisjoint
   *  @brief Comparison operator for pointer ranges
   *
   *  'a' is RangeLessAndDisjoint from 'b' if its upper bound is <=
   *  the lower bound of 'b', and, because we consider zero-size
   *  ranges to be valid, if its lower bound is strictly less than the
   *  lower bound of 'b'. A consequence of this definition is that two
   *  ranges will be "equivalent", by the STL's definition of the
   *  concept, if and only if they overlap. Thus, using this as the
   *  `compare` operator in an associative map keyed on ranges [a,b),
   *  a<=b (with the equality case denoting a valid but zero-sized
   *  range) allows us to quickly identify overlapping ranges.
   *
   *  This provides benefits to our use-case in two ways. First,
   *  overlapping regions are forbidden. Thus, we will never add a
   *  range that overlaps a previously added range because the new key
   *  will present as equivalent to an existing key. Second, we can
   *  search for pointers p efficiently, using `key_type{p,p}` as the
   *  key. Searching this way will yield a range containing `p`, if
   *  one exists. I have included comparison operators that take a
   *  single pointer to facilitate this computation directly. Because
   *  they operate exactly "as though" we had passed a zero-size
   *  range, the ordering remains consistent and searches maintain
   *  their logarithmic complexity.
   */
  struct RangeLessAndDisjoint
  {
    /** @brief Needed to enable the templated overloads to find,
     *         contains, etc.
     */
    typedef std::true_type is_transparent;

    bool operator()(KeyT const& a, KeyT const& b) const noexcept
    {
      return a.second <= b.first && a.first != b.first;
    }

    bool operator()(void const* const a, KeyT const& b) const noexcept
    {
      return a < b.first;
    }

    bool operator()(KeyT const& a, void const* const b) const noexcept
    {
      return a.first != b && a.second <= b;
    }
  };

private:
  using MapType = std::map<KeyT, Allocator*, RangeLessAndDisjoint>;
  MapType m_registry;
  mutable std::mutex m_registry_mtx;
};  // struct PointerRegistry

LBANNV2_EXPORT PointerRegistry& pointer_registry();

}  // namespace lbannv2
