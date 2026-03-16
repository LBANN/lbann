// NOTE: this file is only compiled when LBANNV2_WITH_MI300A or
// LBANNV2_UNKNOWN_MI300A, so the "#else" clauses below are really
// "#elif LBANNV2_UNKNOWN_MI300A".
#include "lbannv2_config.h"

#include <lbannv2/memory/mi300a_allocator.hpp>
#include <lbannv2/ops/nonzero.hpp>
#include <lbannv2/ops/scalar.hpp>
#include <lbannv2/utils/gpu_utils.hpp>

#include <torch/extension.h>
#include <torch/library.h>

namespace
{

at::Scalar lbannv2__local_scalar_dense_cuda(at::Tensor const& self)
{
#if LBANNV2_WITH_MI300A
  return lbannv2::local_scalar_dense_hip(self);
#else
  if (lbannv2::gpu::is_integrated())
    return lbannv2::local_scalar_dense_hip(self);
  return at::native::_local_scalar_dense_cuda(self);
#endif
}

at::Tensor lbannv2_nonzero(at::Tensor const& self)
{
#if LBANNV2_WITH_MI300A
  return lbannv2::nonzero(self);
#else
  if (lbannv2::gpu::is_integrated())
    return lbannv2::nonzero(self);
  return at::native::nonzero_cuda(self);
#endif
}

at::Tensor& lbannv2_nonzero_out(at::Tensor const& self, at::Tensor& out)
{
#if LBANNV2_WITH_MI300A
  return lbannv2::nonzero_out(self, out);
#else
  if (lbannv2::gpu::is_integrated())
    return lbannv2::nonzero_out(self, out);
  return at::native::nonzero_out_cuda(self, out);
#endif
}

} // namespace

TORCH_LIBRARY_IMPL(aten, CUDA, m)
{
  m.impl("_local_scalar_dense", TORCH_FN(lbannv2__local_scalar_dense_cuda));
  m.impl("nonzero", TORCH_FN(lbannv2_nonzero));
  m.impl("nonzero.out", TORCH_FN(lbannv2_nonzero_out));
}
