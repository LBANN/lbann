////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include "lbannv2_config.h"

#include "fallback.hpp"

#include "lbannv2/backend/library_state.hpp"
#include "lbannv2/memory/allocator.hpp"
#include "lbannv2/memory/memory_utils.hpp"
#include "lbannv2/ops/empty_tensor.hpp"
#include "lbannv2/utils/device_helpers.hpp"
#include "lbannv2/utils/logging.hpp"
#include "lbannv2/utils/tensor_helpers.hpp"

#include <ATen/core/ivalue.h>

namespace
{

// Note: PRIVATE so I'm not putting much effort into the name here...
template <typename T>
concept OkTensor =
  std::is_same_v<T, at::Tensor> || std::is_same_v<T, std::optional<at::Tensor>>;

template <OkTensor T>
std::vector<T>
alias_tensor_list_to_real_device(std::vector<T> const& tensor_list)
{
  std::vector<T> out;
  out.reserve(tensor_list.size());
  for (auto const& t : tensor_list)
    out.emplace_back(lbannv2::alias_as_native_device(t));
  return out;
}

bool any_defined(c10::List<at::Tensor> const& tl)
{
  for (auto const& i : c10::irange(tl.size()))
    if (tl[i].defined())
      return true;
  return false;
}

[[maybe_unused]] at::Tensor reset_to_lbann(at::Tensor const& t)
{
  return lbannv2::alias_as_device(
    t, lbannv2::to_lbann(t.device()), t.key_set().add(lbannv2::LBANNDispKey));
}

at::Tensor deepcopy_to_lbann(at::Tensor const& t)
{
  at::Tensor out = lbannv2::empty_lbann(t.sizes(),
                                        t.dtype().toScalarType(),
                                        t.layout(),
                                        std::nullopt,
                                        t.is_pinned(),
                                        std::nullopt);
  out.copy_(t);
  return out;
}

void opt_sync_data_ptr_device(std::optional<at::Tensor> const& t)
{
  if (t)
    lbannv2::sync_data_ptr_device(*t);
}

void check_and_set_device(std::optional<c10::Device>& device_opt,
                          c10::Device const& d)
{
  if (device_opt.has_value() && *device_opt != d)
  {
    LBANNV2_WARN(
      "Detected different devices (d1={}, d2={})", device_opt->str(), d.str());
    // LBANNV2_ASSERT(*device_opt == d,
    //                std::runtime_error,
    //                "LBANN does not handle multi-device kernels at this
    //                time.");
  }
  else
    device_opt = d;
}

}  // namespace

// A few gotchas I have seen allusions to:
//
//   1. In order for zero-copy to be "correct" memory-wise, we need to
//      make sure that any returns get pre-allocated.
//
//   2. Any fallback call may not properly allocate intermediaries
//      with our allocator. We should look at ways to possibly
//      intercept those, too.
//
// Need to think about:
//
//   -> How do we carry stream information?
//   -> How do we carry h2::Device information?
void lbannv2::lbannv2_fallback(c10::OperatorHandle const& op,
                               c10::DispatchKeySet ks,
                               torch::jit::Stack* stack)
{
  // The schema carries a lot of information about the function being
  // dispatched, including detailed information about arguments and
  // return value(s).
  auto const& schema = op.schema();
  auto const& schema_args = schema.arguments();
  auto const num_args = schema_args.size();
  auto const args_beg = stack->size() - num_args;

  LBANNV2_DEBUG("lbannv2_fallback(schema=\"{}\", keyset={})",
                c10::toString(schema),
                c10::toString(ks));

  // Cache the original tensors
  std::vector<at::Tensor> orig_tensors, alias_tensors;
  std::vector<unsigned long> orig_tensor_idx;

  std::vector<c10::List<at::Tensor>> orig_tensor_lists;
  std::vector<unsigned long> orig_tensor_list_idx;

  std::vector<c10::List<std::optional<at::Tensor>>> orig_optional_tensor_lists;
  std::vector<unsigned long> orig_optional_tensor_list_idx;

  std::optional<c10::Device> underlying_device {std::nullopt};

  // Find tensors and device parameters
  auto args = torch::jit::last(*stack, num_args);
  for (auto const i : c10::irange(num_args))
  {
    if (args[i].isTensor())
    {
      auto& tensor_arg = orig_tensors.emplace_back(args[i].toTensor());
      orig_tensor_idx.push_back(i);
      auto& alias =
        alias_tensors.emplace_back(alias_as_native_device(tensor_arg));

      if (tensor_arg.defined())
      {
        LBANNV2_TRACE("  args \"{}\": tensor={}, device={}, alias_device={}",
                      schema_args[i].name(),
                      to_str(tensor_arg),
                      tensor_arg.device().str(),
                      alias.device().str());
        if (!is_scalar(tensor_arg))
          check_and_set_device(underlying_device, alias.device());
      }
      else
      {
        LBANNV2_TRACE("  args \"{}\": undefined", schema_args[i].name());
      }
      // FIXME (trb): What happens when the tensor is not defined??
      (*stack)[args_beg + i] = c10::IValue(alias);
    }
    else if (args[i].isTensorList())
    {
      orig_tensor_lists.emplace_back(args[i].toTensorList());
      orig_tensor_list_idx.push_back(i);
      auto real_dev_tensor_list =
        alias_tensor_list_to_real_device(args[i].toTensorVector());
      // FIXME (trb): Flesh out the same logging and scalar logic as above
      for (auto const& t : real_dev_tensor_list)
        check_and_set_device(underlying_device, t.device());
      (*stack)[args_beg + i] = c10::IValue(std::move(real_dev_tensor_list));
    }
    else if (args[i].isOptionalTensorList())
    {
      orig_optional_tensor_lists.emplace_back(args[i].toOptionalTensorList());
      orig_optional_tensor_list_idx.push_back(i);
      auto real_dev_optional_tensor_list =
        alias_tensor_list_to_real_device(args[i].toOptionalTensorVector());

      size_t otl_idx = 0;
      for (auto const& t : real_dev_optional_tensor_list)
      {
        if (t && t->defined())
        {
          LBANNV2_TRACE(
            "  args \"{}[{}]\": tensor={}, device={}, alias_device={}",
            schema_args[i].name(),
            otl_idx,
            to_str(*t),
            orig_optional_tensor_lists.back().get(otl_idx)->device().str(),
            t->device().str());

          check_and_set_device(underlying_device, t->device());
        }
        ++otl_idx;
      }

      (*stack)[args_beg + i] =
        c10::IValue(std::move(real_dev_optional_tensor_list));
    }
    else if (args[i].isDevice())
    {
      throw std::runtime_error(
        "FIXME (trb): Device arguments are not supported yet.");
      check_and_set_device(underlying_device, args[i].toDevice());
    }
  }

  if (!underlying_device.has_value())
  {
    underlying_device = lbannv2::state::current_device_native();
    LBANNV2_DEBUG("lbann_fallback: op={} using library state device={}.",
                  schema.operator_name().name,
                  underlying_device->str());
  }

  // Push our allocator, redispatch, pop our allocator.
  //
  // NOTE (trb): If multiple underlying devices are detected,
  // 'check_and_set_device` will throw above.
  {
    c10::Device const underlying_dev = *underlying_device;
    auto const underlying_dev_type = underlying_dev.type();
    auto* const orig_alloc = c10::GetAllocator(underlying_dev_type);

    AllocatorWrapper alloc_wrapper(get_allocator(to_lbann(underlying_dev)),
                                   underlying_dev);
    // NOTE (trb): This function's documentation notes that it is not
    // thread-safe and that it is intended for use during
    // initialization. However, it's the only function available for
    // doing this... So here we are... Maybe we should make it an
    // initialization choice to take over the native PyTorch
    // allocators? IDK.
    c10::SetAllocator(underlying_dev_type, &alloc_wrapper);

    // Call the operator again with modified inputs and an updated
    // dispatch keyset.
    op.redispatchBoxed(ks.remove_backend(LBANNBit),
                       stack);  // Seems ok

    c10::SetAllocator(underlying_dev_type, orig_alloc);
  }

  // Now we need to restore stuff. We need to make sure that any
  // tensor allocated by LBANN leaves here with both the tensor and
  // the underlying storage registered with the proper device
  // (PrivateUse1). We should also restore the PrivateUse1 bits to any
  // DispatchKeySet that leaves here associated with LBANN memory.
  //
  // First, let's revert the storage of each input tensor to be back
  // on its original device:
  for (auto const& t : orig_tensors)
    sync_data_ptr_device(t);

  // And also the tensor lists:
  for (auto const& tl : orig_tensor_lists)
    for (auto const& t : tl)
      sync_data_ptr_device(t);

  // And also the optional tensor lists:
  for (auto const& tl : orig_optional_tensor_lists)
    for (auto const& maybe_t : tl)
      opt_sync_data_ptr_device(maybe_t);

  // Now we need to flip through the returns and synchronize alias
  // usage. Writeable aliases must have their metadata resync'd with
  // the output. Non-writeable aliases need to be made to have the
  // proper device/dispatch keys.

  auto const& schema_outs = schema.returns();
  auto const num_outs = schema_outs.size();
  auto const outs_begin = stack->size() - num_outs;
  auto outs = torch::jit::last(*stack, num_outs);
  for (auto const out_idx : c10::irange(num_outs))
  {
    c10::AliasInfo const* const alias_info = schema_outs[out_idx].alias_info();
    if (alias_info)
    {
      bool found = false;
      if (outs[out_idx].isTensor() && outs[out_idx].toTensor().defined())
      {
        for (auto const tensor_idx : c10::irange(orig_tensor_idx.size()))
        {
          auto in_idx = orig_tensor_idx[tensor_idx];
          auto& in_tensor = orig_tensors[tensor_idx];
          auto* in_alias_info = schema_args[in_idx].alias_info();
          if (in_tensor.defined()
              && (in_alias_info == alias_info
                  || (in_alias_info && *in_alias_info == *alias_info)))
          {
            auto orig_out_tensor = outs[out_idx].toTensor();
            if (alias_info->isWrite())
            {
              sync_metadata(orig_out_tensor, in_tensor);
              (*stack)[outs_begin + out_idx] = c10::IValue(in_tensor);
            }
            else
            {
              // This is a non-writeable alias.
              (*stack)[outs_begin + out_idx] =
                c10::IValue(reset_to_lbann(orig_out_tensor));
            }
            found = true;
            break;
          }
        }
      }
      else if (outs[out_idx].isTensorList()
               && any_defined(outs[out_idx].toTensorList()))
      {
        LBANNV2_ASSERT(false,
                       std::runtime_error,
                       "FIXME (trb): Handle tensor list alias returns.");
      }
      LBANNV2_ASSERT(found, std::runtime_error, "Alias mismatch");
    }
    // This is NOT an alias, but it IS a tensor.
    else if (outs[out_idx].isTensor())
    {
      auto out_tensor = outs[out_idx].toTensor();
      // If the output is NOT privateuse1, we need to deep copy
      if (out_tensor.defined() && !is_lbann(out_tensor))
      {
        if (is_managed_ptr(out_tensor.data_ptr()))
          // We own the memory so we just reset the tensor to LBANN
          (*stack)[outs_begin + out_idx] =
            c10::IValue(reset_to_lbann(out_tensor));
        else
          // We don't own the memory, so let's deep copy to LBANN
          (*stack)[outs_begin + out_idx] =
            c10::IValue(deepcopy_to_lbann(out_tensor));
      }
    }
    else if (outs[out_idx].isTensorList())
    {
      LBANNV2_ASSERT(
        false, std::runtime_error, "FIXME (trb): Handle tensor list returns.");
    }
    else if (outs[out_idx].isOptionalTensorList())
    {
      LBANNV2_ASSERT(false,
                     std::runtime_error,
                     "FIXME (trb): Handle optional tensor list returns.");
    }
  }

  LBANNV2_DEBUG("END lbannv2_fallback(op={})", schema.operator_name().name);
}
