////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include "lbannv2/ops/nonzero.hpp"
#include <lbannv2/memory/allocator.hpp>
#include <lbannv2/utils/device_helpers.hpp>
#include <lbannv2/utils/logging.hpp>

#include <h2/gpu/runtime.hpp>

#include <ATen/hip/EmptyTensor.h>
#include <ATen/hip/HIPContext.h>
#include <hipcub/hipcub.hpp>

namespace
{
template <typename T>
T const* get_const(c10::DataPtr const& ptr)
{
  return static_cast<T const*>(ptr.get());
}

// Hoisted from PyTorch; clang-format to LBANNv2's style.
//
//   path: aten/src/ATen/native/cuda/Nonzero.cu
//   commit: 36eb64d60ea6371e3a617ba5026d27be7f88a6af
//
// FIXME: Point to <pytorch>/LICENSE or copy thereof.

template <typename T>
struct NonZeroOp
{
  __host__ __device__ __forceinline__ bool operator()(T const& a) const
  {
    return (a != T(0));
  }
};

#define MAX_DIMS 16
template <typename index_t>
struct TensorDims
{
  index_t sizes[MAX_DIMS];
};

template <typename index_t>
__global__ void write_indices(int64_t* inp,
                              TensorDims<index_t> dims,
                              int ndim,
                              index_t n,
                              int64_t* total = nullptr,
                              int64_t fill_value = -1)
{
  auto index = threadIdx.x + (int64_t) blockIdx.x * blockDim.x;
  bool cond = (total == nullptr || index < *total);
  if (index < n && cond)
  {
    index_t div = 1;
    int64_t idx_flat = inp[index];
#pragma unroll
    for (int dim = MAX_DIMS; dim >= 0; dim--)
    {
      if (dim > ndim - 1)
        continue;
      auto dim_size = dims.sizes[dim];
      inp[index + dim * n] = (idx_flat / div) % dim_size;
      div *= dim_size;
    }
  }
  else if (index < n)
  {
    // 0th dim has correct values already
    for (int dim = ndim - 1; dim > 0; dim--)
    {
      inp[index + dim * n] = fill_value;
    }
  }
}

// Majority from PyTorch. We removed use of host-based
// pinned_num_nonzeros_h. Instead we just sync the stream and use the
// memory directly on the CPU. We also change
// `const_data_ptr<scalar_t>()` to `static_cast<scalar_t
// const*>(const_data_ptr())` to sidestep a linker error with
// amdclang++.
template <typename scalar_t>
void nonzero_out_mi300a_impl(at::Tensor const& self, at::Tensor& out)
{
  at::Tensor self_ = self.contiguous();
  hipStream_t const stream = at::hip::getCurrentHIPStream();
  int64_t chunk_size, num_chunks;
  if (self.numel() < std::numeric_limits<int>::max())
  {
    chunk_size = self.numel();
    num_chunks = 1;
  }
  else
  {
    chunk_size = std::numeric_limits<int>::max() / 2 + 1;  // 2**30
    num_chunks = (self.numel() + chunk_size - 1) / chunk_size;
  }
  // compute number of nonzero elements
  size_t temp_storage_bytes = 0;
  auto& allocator = lbannv2::get_allocator(lbannv2::to_lbann(self.device()));
  auto num_nonzeros = allocator.allocate(sizeof(int) * num_chunks);
  for (int64_t idx = 0; idx < num_chunks; idx++)
  {
    int64_t remaining = std::min(chunk_size, self.numel() - idx * chunk_size);
    hipcub::TransformInputIterator<bool, NonZeroOp<scalar_t>, scalar_t const*>
      itr(static_cast<scalar_t const*>(self_.const_data_ptr()) + idx * chunk_size,
          NonZeroOp<scalar_t>());
    AT_CUDA_CHECK(hipcub::DeviceReduce::Sum(nullptr,
                                            temp_storage_bytes,
                                            itr,
                                            ((int*) num_nonzeros.get()) + idx,
                                            remaining,
                                            stream));
    auto temp_storage = allocator.allocate(temp_storage_bytes);
    AT_CUDA_CHECK(hipcub::DeviceReduce::Sum(temp_storage.get(),
                                            temp_storage_bytes,
                                            itr,
                                            ((int*) num_nonzeros.get()) + idx,
                                            remaining,
                                            stream));
  }

  // TOM: Skip the copy...

  // auto pinned_num_nonzeros_h = at::detail::empty_cpu(
  //     {num_chunks}, /* size */
  //     c10::CppTypeToScalarType<int>(), /* dtype */
  //     std::nullopt, /* layout */
  //     std::nullopt, /* device */
  //     true, /* pin_memory */
  //     std::nullopt /* memory format */
  // );
  // at::cuda::memcpy_and_sync(
  //     (void*)pinned_num_nonzeros_h.const_data_ptr<int>(),
  //     num_nonzeros.get(),
  //     sizeof(int) * num_chunks,
  //     cudaMemcpyDeviceToHost,
  //     stream);

  // TOM: ...just sync the stream...
  H2_CHECK_HIP(hipStreamSynchronize(stream));

  int64_t num_nonzeros_h = 0;

  // TOM: ...and use the pointer.
  for (int64_t idx = 0; idx < num_chunks; idx++)
  {
    num_nonzeros_h += (int) *(get_const<int>(num_nonzeros) + idx);
  }

  // num_nonzeros_h = (int)*(pinned_num_nonzeros_h.const_data_ptr<int>());
  // expected output size is num_nonzeros x ndim
  // we are producing output with size {num_nonzeros, ndim} and strides {1,
  // num_nonzeros} (that is, transposed ndim x num_nonzeros output) we are able
  // to directly use passed output with this size and strides, and we can also
  // (per contract) resize passed output with incorrect sizes anyway we want.
  // However, out with correct sizes and incorrect strides will have to be
  // copied to from the intermediate we've produced.
  bool need_to_copy = out.dim() == 2 && out.sizes()[0] == num_nonzeros_h
                      && out.sizes()[1] == self.dim()
                      && !out.t().is_contiguous();
  at::Tensor out_temp = need_to_copy
                          ? at::Tensor(at::detail::empty_cuda(
                              {self.dim(), num_nonzeros_h}, out.options()))
                          : out.resize_({self.dim(), num_nonzeros_h});
  // Scalars are expected to produce output of size (1,0), so we can't write to
  // it
  int64_t curr_nonzeros = 0;
  if (self.dim() > 0)
  {
    for (int64_t idx = 0; idx < num_chunks; idx++)
    {
      int remaining = std::min(chunk_size, self.numel() - idx * chunk_size);

      hipcub::CountingInputIterator<int64_t> counting_itr(idx * chunk_size);
      hipcub::TransformInputIterator<bool, NonZeroOp<scalar_t>, scalar_t const*>
        itr(static_cast<scalar_t const*>(self_.const_data_ptr()) + idx * chunk_size,
            NonZeroOp<scalar_t>());
      temp_storage_bytes = 0;
      AT_CUDA_CHECK(
        hipcub::DeviceSelect::Flagged(nullptr,
                                      temp_storage_bytes,
                                      counting_itr,
                                      itr,
                                      out_temp.mutable_data_ptr<int64_t>(),
                                      ((int*) num_nonzeros.get()) + idx,
                                      remaining,
                                      stream));
      auto temp_storage = allocator.allocate(temp_storage_bytes);
      AT_CUDA_CHECK(hipcub::DeviceSelect::Flagged(
        temp_storage.get(),
        temp_storage_bytes,
        counting_itr,
        itr,
        out_temp.mutable_data_ptr<int64_t>() + curr_nonzeros,
        ((int*) num_nonzeros.get()) + idx,
        remaining,
        stream));
      // TOM: Oh look, we use it again.
      curr_nonzeros += (int) *(get_const<int>(num_nonzeros) + idx);
    }
    if (num_nonzeros_h > 0 && self.dim() > 1)
    {
      TensorDims<int64_t> dims;
      for (int i = 0; i < self.dim(); i++)
      {
        dims.sizes[i] = self.sizes()[i];
      }
      int const nthreads = 256;
      int const nblocks = (num_nonzeros_h + nthreads - 1) / nthreads;
      write_indices<<<nblocks, nthreads, 0, stream>>>(
        out_temp.mutable_data_ptr<int64_t>(), dims, self.dim(), num_nonzeros_h);
      C10_HIP_KERNEL_LAUNCH_CHECK();
    }
  }
  if (need_to_copy)
  {
    out.copy_(out_temp.t());
  }
  else
  {
    // transpose out so it is correct size
    at::Tensor out_ = out_temp.t();
    out.set_(out_);
  }
}
}  // namespace

at::Tensor& lbannv2::nonzero_out(at::Tensor const& self, at::Tensor& out)
{
  c10::ScalarType const dtype = self.scalar_type();

  LBANNV2_TRACE("lbannv2::nonzero_out(device={}, dtype={})",
                self.device().str(),
                c10::toString(dtype));

  switch (dtype)
  {
  case c10::ScalarType::Bool: nonzero_out_mi300a_impl<bool>(self, out); break;
  case c10::ScalarType::Float:  nonzero_out_mi300a_impl<float>(self, out); break;
  case c10::ScalarType::Double: nonzero_out_mi300a_impl<double>(self, out); break;
  case c10::ScalarType::Int:  nonzero_out_mi300a_impl<int>(self, out); break;
  case c10::ScalarType::UInt32: nonzero_out_mi300a_impl<std::uint32_t>(self, out); break;
  case c10::ScalarType::Long:  nonzero_out_mi300a_impl<long>(self, out); break;
  default: return at::native::nonzero_out_cuda(self, out);
  }

  return out;
}

at::Tensor lbannv2::nonzero(at::Tensor const& self)
{
  at::Tensor out =
    at::detail::empty_cuda({0}, self.options().dtype(c10::kLong));
  return nonzero_out(self, out);
}
