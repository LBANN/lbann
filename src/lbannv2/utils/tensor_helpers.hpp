////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "lbannv2/utils/device_helpers.hpp"
#include "lbannv2/utils/errors.hpp"

#include <ATen/NamedTensorUtils.h>
#include <ATen/Tensor.h>
#include <c10/util/ArrayRef.h>

namespace lbannv2
{

/** @brief Determines if t is associated with LBANN */
inline bool is_lbann(at::Tensor const& t)
{
  return t.is_privateuseone();
}

inline bool is_scalar(at::Tensor const& t)
{
  return t.defined() && (t.dim() == 0);
}

inline c10::Device get_underlying_device(at::Tensor const& t)
{
  return to_native(t.device());
}

inline void set_data_ptr_device(c10::DataPtr& dp, c10::Device d)
{
  dp.unsafe_set_device(std::move(d));
}

inline void set_data_ptr_device(c10::Storage const& s, c10::Device d)
{
  set_data_ptr_device(s.mutable_data_ptr(), std::move(d));
}

inline void set_data_ptr_device(at::Tensor const& t, c10::Device d)
{
  set_data_ptr_device(t.storage(), std::move(d));
}

inline void sync_metadata(at::Tensor const& src, at::Tensor& dst)
{
  auto* dst_tensor_info = dst.unsafeGetTensorImpl();
  dst_tensor_info->set_storage_offset(src.storage_offset());
  dst_tensor_info->set_sizes_and_strides(src.sizes(), src.strides());

  // I assume this restores named dimensions? Not sure if it
  // should be here or not. See "alias_with_sizes_and_strides"
  // in <pytorch>/aten/src/ATen/native/TensorShape.cpp
  at::namedinference::propagate_names(dst, src);
}

/** @brief Make an alias of the tensor on a new backend
 *
 *  This function can be used to produce aliases with diffent devices,
 *  different dispatch keys, or both (or neither, I suppose).
 *
 *  @post The original tensor will keep its device type and keys, but
 *        its DataPtr will appear to be on the new device if queried.
 */
inline at::Tensor alias_as_device(at::Tensor const& orig_tensor,
                                  c10::Device const& d,
                                  c10::DispatchKeySet ks)
{
  // Make (soft) copy of the storage and set the device to be the real
  // underlying device.
  at::Storage aliased_storage(orig_tensor.storage());
  set_data_ptr_device(aliased_storage, d);

  // Set up a view with this storage, using the modified keyset.
  auto alias_tensor =
    at::detail::make_tensor<at::TensorImpl>(c10::TensorImpl::VIEW,
                                            std::move(aliased_storage),
                                            std::move(ks),
                                            orig_tensor.dtype());

  // Setup sizes, strides, and storage offset.
  sync_metadata(orig_tensor, alias_tensor);

  // Quick sanity check before we go
  LBANNV2_ASSERT(alias_tensor.const_data_ptr() == orig_tensor.const_data_ptr(),
                 std::runtime_error,
                 "Aliasing tensor data has failed");

  return alias_tensor;
}

/** @brief Alias the tensor to the underlying device.
 *
 *  This effectively removes the LBANN/PrivateUse1 bits from the
 *  tensor's metadata. If the tensor is not an LBANN tensor to begin
 *  with, it just returns (a soft copy) of the input tensor.
 *
 *  @post The original tensor will keep its device type and keys, but
 *        its DataPtr will appear to be on the underlying device if
 *        queried.
 */
inline at::Tensor alias_as_native_device(at::Tensor const& orig_tensor)
{
  if (!is_lbann(orig_tensor))
    return orig_tensor;

  // Get the original device (should be 'lbann'/'privateuseone') and
  // the underlying device where the memory resides
  // ('cpu'/'cuda'/etc); remove PrivateUse1 from the dispatch keyset.
  return alias_as_device(orig_tensor,
                         get_underlying_device(orig_tensor),
                         orig_tensor.key_set().remove_backend(LBANNBit));
}

inline std::optional<at::Tensor>
alias_as_native_device(std::optional<at::Tensor> const& t)
{
  if (t.has_value())
    return alias_as_native_device(t.value());
  return std::nullopt;
}

/** @brief Set the underlying DataPtr to the same device as input
 *
 *  @post `t.storage().data_ptr().device() == t.device()`
 */
inline void sync_data_ptr_device(at::Tensor const& t)
{
  if (t.defined())
    set_data_ptr_device(t, t.device());
}

/** @brief Minimal tensor stringification.
 *
 *  Returns "[ {device type}{data type}[d1, d2, ..., dn] ]", for
 *  example, "[ lbannFloatType[2, 2] ]" for a 2x2 Float32 tensor on
 *  the LBANN backend.
 */
inline std::string to_str(at::Tensor const& t)
{
  std::ostringstream oss;
  oss << "[ " << t.toString() << t.sizes() << " ]";
  return oss.str();
}

/** @brief ArrayRef stringification */
template <typename T>
std::string to_str(c10::ArrayRef<T> const& ar)
{
  std::ostringstream oss;
  oss << ar;
  return oss.str();
}

}  // namespace lbannv2
