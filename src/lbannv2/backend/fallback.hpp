////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

// FIXME (trb): I think this file could live either in backend/ or in
// ops/ (since this is the "fallback operator"). However, I think it's
// more a 'backend requirement' thing rather than a *specific* op, so
// I've put it in backend/ for now. Agree? Disagree?

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <c10/core/DispatchKeySet.h>

namespace lbannv2
{

/** @brief A default dispatch fallback for the LBANN backend
 *
 *  This function provides a fallback capability for LBANN by
 *  attempting to alias all LBANN tensor arguments the device type
 *  that matches their memory residency. That is, LBANN-owned
 *  CPU-allocated tensors will appear to be associated with the "CPU"
 *  backend while LBANN-owned CUDA/HIP-allocated tensors will appear
 *  to be associated with the "CUDA" backend, etc. Non-LBANN-owned
 *  tensors should not be modified at all. Tensor return values that
 *  are not aliases of input data are not supported at this time
 *  (there could be ambiguity in the semantics that prevents deciding
 *  which of any input devices is the correct output device).
 *
 *  This should be robust to the set of kernels that LBANN does not
 *  implement directly. If errors or missing kernels are found, please
 *  report it by opening an issue.
 *
 *  Interesting things can happen depending on how the underlying
 *  implementation calls into dispatched operators. In particular,
 *  there can be a significant difference between dispatching a
 *  "high-level" kernel to a native backend and making a sequence of
 *  "low-level" dispatched calls, the former possibly avoiding such a
 *  sequence altogether.
 *
 *  See https://dev-discuss.pytorch.org/t/backend-fallbacks/195 for a
 *  general discussion of backend fallbacks.
 */
void lbannv2_fallback(c10::OperatorHandle const& op,
                      c10::DispatchKeySet ks,
                      torch::jit::Stack* stack);

}  // namespace lbannv2
