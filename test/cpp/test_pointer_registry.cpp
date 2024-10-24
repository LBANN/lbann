////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include <lbannv2/memory/registry.hpp>
#include <lbannv2/utils/device_helpers.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

TEST_CASE("RangeLessAndDisjoint", "[memory][registry]")
{
  std::vector<unsigned char> buffer(8);
  lbannv2::PointerRegistry::RangeLessAndDisjoint rng_less;

  SECTION("Non-overlapping ranges behave sanely")
  {
    CHECK(rng_less({&buffer[1], &buffer[2]}, {&buffer[3], &buffer[4]}));
    CHECK_FALSE(rng_less({&buffer[3], &buffer[4]}, {&buffer[1], &buffer[2]}));
  }

  SECTION("Abutting ranges are nonoverlapping")
  {
    CHECK(rng_less({&buffer[1], &buffer[2]}, {&buffer[2], &buffer[3]}));
    CHECK_FALSE(rng_less({&buffer[2], &buffer[3]}, {&buffer[1], &buffer[2]}));
  }

  SECTION("Identical ranges")
  {
    CHECK_FALSE(rng_less({&buffer[1], &buffer[4]}, {&buffer[1], &buffer[4]}));
  }

  SECTION("Partially overlapping ranges")
  {
    CHECK_FALSE(rng_less({&buffer[1], &buffer[4]}, {&buffer[2], &buffer[5]}));
    CHECK_FALSE(rng_less({&buffer[2], &buffer[5]}, {&buffer[1], &buffer[4]}));
  }

  SECTION("One range proper subset of the other")
  {
    CHECK_FALSE(rng_less({&buffer[1], &buffer[8]}, {&buffer[3], &buffer[4]}));
    CHECK_FALSE(rng_less({&buffer[3], &buffer[4]}, {&buffer[1], &buffer[8]}));
  }

  SECTION("Zero-size ranges work appropriately")
  {
    CHECK(rng_less({&buffer[1], &buffer[1]}, {&buffer[2], &buffer[2]}));
    CHECK_FALSE(rng_less({&buffer[2], &buffer[2]}, {&buffer[1], &buffer[1]}));

    CHECK(rng_less({&buffer[1], &buffer[1]}, {&buffer[2], &buffer[4]}));
    CHECK_FALSE(rng_less({&buffer[2], &buffer[4]}, {&buffer[1], &buffer[1]}));

    CHECK(rng_less({&buffer[1], &buffer[2]}, {&buffer[2], &buffer[2]}));
    CHECK_FALSE(rng_less({&buffer[2], &buffer[2]}, {&buffer[1], &buffer[2]}));

    CHECK(rng_less({&buffer[1], &buffer[2]}, &buffer[2]));
    CHECK(rng_less(&buffer[1], {&buffer[2], &buffer[3]}));

    CHECK_FALSE(rng_less(&buffer[1], {&buffer[1], &buffer[2]}));
    CHECK_FALSE(rng_less({&buffer[1], &buffer[2]}, &buffer[1]));

    CHECK_FALSE(rng_less(&buffer[1], {&buffer[1], &buffer[1]}));
    CHECK_FALSE(rng_less({&buffer[1], &buffer[1]}, &buffer[1]));
  }
}

namespace
{
size_t rng_bytes(std::pair<void*, void*> const& r)
{
  return std::distance((std::byte*) r.first, (std::byte*) r.second);
}
}  // namespace

TEST_CASE("PointerRegistry::add()", "[memory][registry]")
{
  using RangeT = std::pair<void*, void*>;

  lbannv2::PointerRegistry registry;
  std::vector<unsigned char> buffer(32);

  // Establish preconditions
  REQUIRE(registry.num_registered() == 0UL);
  REQUIRE(registry.bytes_registered() == 0UL);

  SECTION("Adding nonoverlapping regions is successful.")
  {
    RangeT const rng1 = {&buffer[4], &buffer[8]};
    RangeT const rng2 = {&buffer[12], &buffer[16]};
    RangeT const rng3 = {&buffer[16], &buffer[20]};
    RangeT const rng4 = {&buffer[8], &buffer[12]};

    size_t expected_bytes = 0UL;
    REQUIRE_NOTHROW(registry.add(rng1.first, rng_bytes(rng1), nullptr));
    expected_bytes += rng_bytes(rng1);

    REQUIRE(registry.num_registered() == 1UL);
    REQUIRE(registry.bytes_registered() == expected_bytes);

    REQUIRE_NOTHROW(registry.add(rng2.first, rng_bytes(rng2), nullptr));
    expected_bytes += rng_bytes(rng2);

    REQUIRE(registry.num_registered() == 2UL);
    REQUIRE(registry.bytes_registered() == expected_bytes);

    REQUIRE_NOTHROW(registry.add(rng3.first, rng_bytes(rng3), nullptr));
    expected_bytes += rng_bytes(rng3);

    REQUIRE(registry.num_registered() == 3UL);
    REQUIRE(registry.bytes_registered() == expected_bytes);

    REQUIRE_NOTHROW(registry.add(rng4.first, rng_bytes(rng4), nullptr));
    expected_bytes += rng_bytes(rng4);

    REQUIRE(registry.num_registered() == 4UL);
    REQUIRE(registry.bytes_registered() == expected_bytes);
  }

  SECTION("Zero-size regions")
  {
    SECTION("Adding zero-size regions is ok.")
    {
      RangeT const rng1 = {&buffer[0], &buffer[0]};
      RangeT const rng2 = {&buffer[2], &buffer[2]};

      REQUIRE_NOTHROW(registry.add(rng1.first, rng_bytes(rng1), nullptr));
      REQUIRE(registry.num_registered() == 1UL);
      REQUIRE(registry.bytes_registered() == 0UL);

      REQUIRE_NOTHROW(registry.add(rng2.first, rng_bytes(rng2), nullptr));
      REQUIRE(registry.num_registered() == 2UL);
      REQUIRE(registry.bytes_registered() == 0UL);
    }

    SECTION("Zero-size regions are not valid start points for other regions")
    {
      RangeT const zero_rng = {&buffer[0], &buffer[0]};
      RangeT const other_rng = {&buffer[0], &buffer[2]};

      REQUIRE_NOTHROW(
        registry.add(zero_rng.first, rng_bytes(zero_rng), nullptr));
      REQUIRE(registry.num_registered() == 1UL);
      REQUIRE(registry.bytes_registered() == 0UL);

      REQUIRE_THROWS_WITH(
        registry.add(other_rng.first, rng_bytes(other_rng), nullptr),
        "Address range overlaps existing range");
    }

    SECTION("Zero-size regions are valid end points for other regions")
    {
      RangeT const other_rng = {&buffer[0], &buffer[2]};
      RangeT const zero_rng = {&buffer[2], &buffer[2]};

      REQUIRE_NOTHROW(
        registry.add(zero_rng.first, rng_bytes(zero_rng), nullptr));
      REQUIRE(registry.num_registered() == 1UL);
      REQUIRE(registry.bytes_registered() == 0UL);

      REQUIRE_NOTHROW(
        registry.add(other_rng.first, rng_bytes(other_rng), nullptr));
      REQUIRE(registry.num_registered() == 2UL);
      REQUIRE(registry.bytes_registered() == rng_bytes(other_rng));
    }
  }
}

TEST_CASE("PointerRegistry::remove()", "[memory][registry]")
{
  using RangeT = std::pair<void*, void*>;

  lbannv2::PointerRegistry registry;
  std::vector<unsigned char> buffer(32);

  // Establish preconditions
  REQUIRE(registry.num_registered() == 0UL);
  REQUIRE(registry.bytes_registered() == 0UL);

  SECTION("Removing a context pointer works")
  {
    RangeT const rng = {&buffer[4], &buffer[8]};
    REQUIRE_NOTHROW(registry.add(rng.first, rng_bytes(rng), nullptr));
    REQUIRE(registry.num_registered() == 1UL);
    REQUIRE(registry.bytes_registered() == rng_bytes(rng));

    REQUIRE_NOTHROW(registry.remove(rng.first));
    REQUIRE(registry.num_registered() == 0UL);
    REQUIRE(registry.bytes_registered() == 0UL);
  }

  SECTION("Removing a known non-context pointer fails")
  {
    RangeT const rng = {&buffer[4], &buffer[8]};
    void* const noncontext_ptr = &buffer[6];

    REQUIRE_NOTHROW(registry.add(rng.first, rng_bytes(rng), nullptr));
    REQUIRE(registry.num_registered() == 1UL);
    REQUIRE(registry.bytes_registered() == rng_bytes(rng));

    REQUIRE_THROWS_WITH(registry.remove(noncontext_ptr),
                        "Cannot remove ptr; not beginning of range.");
    REQUIRE(registry.num_registered() == 1UL);
    REQUIRE(registry.bytes_registered() == rng_bytes(rng));
  }

  SECTION("Removing an unknown pointer fails")
  {
    RangeT const rng = {&buffer[4], &buffer[8]};
    void* const unknown_ptr = &buffer[16];

    REQUIRE_NOTHROW(registry.add(rng.first, rng_bytes(rng), nullptr));
    REQUIRE(registry.num_registered() == 1UL);
    REQUIRE(registry.bytes_registered() == rng_bytes(rng));

    REQUIRE_THROWS_AS(registry.remove(unknown_ptr), lbannv2::UnknownAddress);
    REQUIRE(registry.num_registered() == 1UL);
    REQUIRE(registry.bytes_registered() == rng_bytes(rng));
  }

  SECTION("Removing a zero-size region is ok")
  {
    RangeT const rng = {&buffer[2], &buffer[2]};
    REQUIRE_NOTHROW(registry.add(rng.first, rng_bytes(rng), nullptr));
    REQUIRE(registry.num_registered() == 1UL);
    REQUIRE(registry.bytes_registered() == 0UL);

    REQUIRE_NOTHROW(registry.remove(rng.first));
    REQUIRE(registry.num_registered() == 0UL);
    REQUIRE(registry.bytes_registered() == 0UL);
  }
}

TEST_CASE("PointerRegistry::known()", "[memory][registry]")
{
  using RangeT = std::pair<void*, void*>;

  lbannv2::PointerRegistry registry;
  std::vector<unsigned char> buffer(32);

  // Establish preconditions
  REQUIRE(registry.num_registered() == 0UL);
  REQUIRE(registry.bytes_registered() == 0UL);

  SECTION("Pointers in registered ranges are known")
  {
    RangeT const rng = {&buffer[4], &buffer[8]};
    void const* const context_ptr = &buffer[4];
    void const* const noncontext_ptr = &buffer[6];
    REQUIRE_NOTHROW(registry.add(rng.first, rng_bytes(rng), nullptr));
    REQUIRE(registry.num_registered() == 1UL);
    REQUIRE(registry.bytes_registered() == rng_bytes(rng));

    REQUIRE(registry.known(context_ptr));
    REQUIRE(registry.known(noncontext_ptr));
  }

  SECTION("Registered pointer in size-zero ranges are known")
  {
    RangeT const rng = {&buffer[4], &buffer[4]};
    void const* const context_ptr = &buffer[4];
    REQUIRE_NOTHROW(registry.add(rng.first, rng_bytes(rng), nullptr));
    REQUIRE(registry.num_registered() == 1UL);
    REQUIRE(registry.bytes_registered() == rng_bytes(rng));

    REQUIRE(registry.known(context_ptr));
  }

  SECTION("Pointers outside registered ranges are not known")
  {
    RangeT const rng = {&buffer[4], &buffer[8]};
    void const* const unknown_low_ptr = &buffer[2];
    void const* const unknown_ub_ptr = &buffer[8];
    void const* const unknown_high_ptr = &buffer[14];

    REQUIRE_NOTHROW(registry.add(rng.first, rng_bytes(rng), nullptr));
    REQUIRE(registry.num_registered() == 1UL);
    REQUIRE(registry.bytes_registered() == rng_bytes(rng));

    REQUIRE_FALSE(registry.known(unknown_low_ptr));
    REQUIRE_FALSE(registry.known(unknown_ub_ptr));
    REQUIRE_FALSE(registry.known(unknown_high_ptr));
  }
}

TEST_CASE("PointerRegistry::get_context()", "[memory][registry]")
{
  using RangeT = std::pair<void*, void*>;

  lbannv2::PointerRegistry registry;
  std::vector<unsigned char> buffer(32);

  // Establish preconditions
  REQUIRE(registry.num_registered() == 0UL);
  REQUIRE(registry.bytes_registered() == 0UL);

  SECTION("Context pointers are their own context")
  {
    RangeT const rng1 = {&buffer[4], &buffer[8]};
    RangeT const rng2 = {&buffer[12], &buffer[16]};
    RangeT const zero_rng = {&buffer[20], &buffer[20]};

    void const* const context_ptr1 = &buffer[4];
    void const* const context_ptr2 = &buffer[12];
    void const* const zero_context_ptr = &buffer[20];

    REQUIRE_NOTHROW(registry.add(rng1.first, rng_bytes(rng1), nullptr));
    REQUIRE_NOTHROW(registry.add(rng2.first, rng_bytes(rng2), nullptr));
    REQUIRE_NOTHROW(registry.add(zero_rng.first, rng_bytes(zero_rng), nullptr));
    REQUIRE(registry.num_registered() == 3UL);
    REQUIRE(registry.bytes_registered()
            == rng_bytes(rng1) + rng_bytes(rng2) + rng_bytes(zero_rng));

    REQUIRE(registry.get_context(context_ptr1) == rng1.first);
    REQUIRE(registry.get_context(context_ptr2) == rng2.first);
    REQUIRE(registry.get_context(zero_context_ptr) == zero_rng.first);
  }

  SECTION("Noncontext pointers return the proper context pointer")
  {
    RangeT const rng1 = {&buffer[4], &buffer[8]};
    RangeT const rng2 = {&buffer[12], &buffer[16]};

    void const* const noncontext_ptr1 = &buffer[6];
    void const* const noncontext_ptr2 = &buffer[14];

    REQUIRE_NOTHROW(registry.add(rng1.first, rng_bytes(rng1), nullptr));
    REQUIRE_NOTHROW(registry.add(rng2.first, rng_bytes(rng2), nullptr));

    REQUIRE(registry.num_registered() == 2UL);
    REQUIRE(registry.bytes_registered() == rng_bytes(rng1) + rng_bytes(rng2));

    REQUIRE(registry.get_context(noncontext_ptr1) == rng1.first);
    REQUIRE(registry.get_context(noncontext_ptr2) == rng2.first);
  }

  SECTION("Unknown pointers fail")
  {
    RangeT const rng1 = {&buffer[4], &buffer[8]};
    RangeT const rng2 = {&buffer[12], &buffer[16]};

    void const* const ptr1 = &buffer[2];
    void const* const ptr2 = &buffer[8];
    void const* const ptr3 = &buffer[10];
    void const* const ptr4 = &buffer[16];
    void const* const ptr5 = &buffer[20];

    REQUIRE_NOTHROW(registry.add(rng1.first, rng_bytes(rng1), nullptr));
    REQUIRE_NOTHROW(registry.add(rng2.first, rng_bytes(rng2), nullptr));

    REQUIRE(registry.num_registered() == 2UL);
    REQUIRE(registry.bytes_registered() == rng_bytes(rng1) + rng_bytes(rng2));

    REQUIRE_THROWS_AS(registry.get_context(ptr1), lbannv2::UnknownAddress);
    REQUIRE_THROWS_AS(registry.get_context(ptr2), lbannv2::UnknownAddress);
    REQUIRE_THROWS_AS(registry.get_context(ptr3), lbannv2::UnknownAddress);
    REQUIRE_THROWS_AS(registry.get_context(ptr4), lbannv2::UnknownAddress);
    REQUIRE_THROWS_AS(registry.get_context(ptr5), lbannv2::UnknownAddress);
  }
}

TEST_CASE("PointerRegistry::unsafe_reset_allocator()", "[memory][registry]")
{
  using RangeT = std::pair<void*, void*>;

  lbannv2::PointerRegistry registry;
  std::vector<unsigned char> buffer(32);

  // Establish preconditions
  REQUIRE(registry.num_registered() == 0UL);
  REQUIRE(registry.bytes_registered() == 0UL);

  RangeT const rng = {&buffer[4], &buffer[8]};

  void const* const ctxt_ptr = &buffer[4];
  void const* const mid_ptr = &buffer[6];
  void const* const bad_ptr = &buffer[0];

  lbannv2::Allocator& alloc =
    lbannv2::get_allocator({lbannv2::LBANNDeviceT, lbannv2::LBANN_CPU});
  lbannv2::Allocator* orig_alloc = &alloc;

  // FAKE -- DO NOT DEREFERENCE!
  lbannv2::Allocator* other_alloc = ++orig_alloc;

  // Get the allocator setup
  REQUIRE_NOTHROW(registry.add(rng.first, rng_bytes(rng), orig_alloc));
  REQUIRE(registry.get_allocator(ctxt_ptr) == orig_alloc);
  REQUIRE(registry.get_allocator(mid_ptr) == orig_alloc);

  SECTION("Resetting by context is ok")
  {
    REQUIRE_NOTHROW(registry.unsafe_reset_allocator(ctxt_ptr, other_alloc));
    REQUIRE(registry.get_allocator(ctxt_ptr) == other_alloc);
    REQUIRE(registry.get_allocator(mid_ptr) == other_alloc);
  }

  SECTION("Resetting by an interior pointer is ok")
  {
    // FIXME: Perhaps this should actually be disallowed??
    REQUIRE_NOTHROW(registry.unsafe_reset_allocator(ctxt_ptr, other_alloc));
    REQUIRE(registry.get_allocator(ctxt_ptr) == other_alloc);
    REQUIRE(registry.get_allocator(mid_ptr) == other_alloc);
  }

  SECTION("Resetting an unknown pointer fails")
  {
    REQUIRE_THROWS_AS(registry.unsafe_reset_allocator(bad_ptr, other_alloc),
                      lbannv2::UnknownAddress);
  }
}

TEST_CASE("PointerRegistry::bytes_registered()", "[memory][registry]")
{
  using RangeT = std::pair<void*, void*>;

  lbannv2::PointerRegistry registry;
  std::vector<unsigned char> buffer(16);

  // Establish preconditions
  REQUIRE(registry.num_registered() == 0UL);
  REQUIRE(registry.bytes_registered() == 0UL);

  RangeT const rng = {&buffer[4], &buffer[8]};
  size_t const rng_size = rng_bytes(rng);

  void const* const ctxt_ptr = &buffer[4];
  void const* const mid_ptr = &buffer[6];
  void const* const extern_ptr_1 = &buffer[0];
  void const* const extern_ptr_2 = &buffer[16];

  REQUIRE_NOTHROW(registry.add(rng.first, rng_bytes(rng), nullptr));
  REQUIRE(registry.bytes_registered() == rng_size);

  CHECK(registry.bytes_registered(ctxt_ptr) == rng_size);
  CHECK(registry.bytes_registered(mid_ptr) == rng_size);
  CHECK(registry.bytes_registered(extern_ptr_1) == 0UL);
  CHECK(registry.bytes_registered(extern_ptr_2) == 0UL);

}
