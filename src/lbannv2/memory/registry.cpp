////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include "registry.hpp"

#include "lbannv2/utils/errors.hpp"
#include "lbannv2/utils/logging.hpp"

namespace
{

// Syntactic sugar. Iterators kinda suck for readability.
auto const& get_ptr_range(std::input_iterator auto const& map_iter) noexcept
{
  return map_iter->first;
}

auto const& get_allocator_ptr(std::input_iterator auto const& map_iter) noexcept
{
  return map_iter->second;
}

std::size_t range_bytes(std::pair<void*, void*> const& r) noexcept
{
  return std::distance((std::byte*) r.first, (std::byte*) r.second);
}

}  // namespace

namespace lbannv2
{

void PointerRegistry::add(void* const ptr,
                          size_t const size,
                          Allocator* const allocator)
{
  std::lock_guard<std::mutex> lock(m_registry_mtx);
  auto const [it, added] = m_registry.emplace(
    KeyT {ptr, static_cast<std::byte*>(ptr) + size}, allocator);
  LBANNV2_ASSERT(
    added, std::runtime_error, "Address range overlaps existing range");

  LBANNV2_TRACE("Registered pointer range start={}, size={}, allocator={}",
                ptr,
                size,
                (void*) allocator);
}

void PointerRegistry::remove(void* const ptr)
{
  std::lock_guard<std::mutex> lock(m_registry_mtx);
  auto const it = m_registry.find(ptr);
  if (it == m_registry.cend())
    throw UnknownAddress {};
  else if (get_ptr_range(it).first != ptr)
    throw std::runtime_error("Cannot remove ptr; not beginning of range.");

  {
    [[maybe_unused]] auto const& [ptr_range, alloc_ptr] = *it;
    LBANNV2_TRACE("Deregistered pointer range start={}, size={}, allocator={}",
                  ptr_range.first,
                  range_bytes(ptr_range),
                  (void*) alloc_ptr);
  }

  m_registry.erase(it);
}

bool PointerRegistry::known(void const* const ptr) const noexcept
{
  std::lock_guard<std::mutex> lock(m_registry_mtx);
  return m_registry.contains(ptr);
}

Allocator* PointerRegistry::get_allocator(void const* const ptr) const
{
  std::lock_guard<std::mutex> lock(m_registry_mtx);
  auto const it = m_registry.find(ptr);
  if (it == m_registry.cend())
    throw UnknownAddress {};
  return get_allocator_ptr(it);
}

void PointerRegistry::unsafe_reset_allocator(void const* const ptr,
                                             Allocator* const new_alloc)
{
  std::lock_guard<std::mutex> lock(m_registry_mtx);
  auto const it = m_registry.find(ptr);
  if (it == m_registry.cend())
    throw UnknownAddress {};
  it->second = new_alloc;
}

void* PointerRegistry::get_context(void const* const ptr) const
{
  std::lock_guard<std::mutex> lock(m_registry_mtx);
  auto const it = m_registry.find(ptr);
  if (it == m_registry.cend())
    throw UnknownAddress {};
  return get_ptr_range(it).first;
}

std::size_t PointerRegistry::bytes_registered() const noexcept
{
  std::lock_guard<std::mutex> lock(m_registry_mtx);
  size_t bytes = 0UL;
  for (auto const& kvp : m_registry)
  {
    bytes += range_bytes(kvp.first);
  }
  return bytes;
}

std::size_t
PointerRegistry::bytes_registered(void const* const ptr) const noexcept
{
  std::lock_guard<std::mutex> lock(m_registry_mtx);
  auto const it = m_registry.find(ptr);
  if (it != m_registry.cend())
  {
    return range_bytes(it->first);
  }
  return 0;
}

}  // namespace lbannv2

auto lbannv2::pointer_registry() -> PointerRegistry&
{
  static PointerRegistry registry;
  return registry;
}
