////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

/** Copy between two device buffers, using all threads in a warp. */
#define LBANN_TRANSFORM_DISTCONV_NVSHMEM_VECTOR_ADDRESSING_INSTANTIATE
#include "lbann/utils/distconv.hpp"
#include "lbann/base.hpp"

#if defined(LBANN_HAS_NVSHMEM) && defined(LBANN_HAS_DISTCONV)
#include "nvshmem.h"
#include "nvshmemx.h"
#include "lbann/layers/transform/distconv/distconv_nvshmem_vector_addressing.hpp"

namespace distconv{

  namespace{

 __device__ __forceinline__
float atomic_add(float* __restrict__ address,
                 const float val,
                 const int pe){

  int* address_as_int = (int*)address;
  int assumed; 
  int old = nvshmem_int_g(address_as_int, pe);
  do
  {
    assumed = old;
    old = nvshmem_int_atomic_compare_swap(address_as_int, assumed,
                                          __float_as_int(val +
                                                         __int_as_float(assumed)),
                                          pe);
  } while (assumed !=old);
  return __int_as_float(old);
}

__device__ __forceinline__ 
double atomic_add(double* __restrict__ address,
                  const double val,
                  const int pe){

  long long int* address_as_ll = (long long int*)address;
  long long int assumed; 
  long long int old = nvshmem_longlong_g(address_as_ll, pe);
  do
  {
    assumed = old;
    old = nvshmem_longlong_atomic_compare_swap(address_as_ll, assumed,
                                               __double_as_longlong(val +
                                                                    __longlong_as_double(assumed)),
                                               pe);
  }while(assumed != old);
  return __longlong_as_double(old);
}

__device__ __forceinline__
float floor(const float& x) {return floorf(x);}

__device__ __forceinline__
double floor(const double& x) {return floor(x);}

  /**
   * @brief NVSHMEM Gather kernel
   * 
   * workspace(k, j, i) = values(k, indices(k, j), i)
   * 
   */

  template <typename DataType>
  __global__ void Gather_NVSHMEM_Kernel(
    const DataType* __restrict__ shared_buffer,
    const DataType* __restrict__ indices,
    DataType* __restrict__ output,
    const int mini_batch_size,
    const int num_local_values_rows,
    const int num_local_cols,
    const int num_local_output_rows,
    const int pe_group,
    const int pe_stride){
    
    constexpr int warp_size = 32;
    // Indice
    const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
    const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
    
    const size_t nthreadsy = gridDim.y * blockDim.y;
    const size_t nthreadsx = gridDim.x * blockDim.x;
    
    for (size_t mb_i = gidy; mb_i < mini_batch_size; mb_i += nthreadsy){
      // Figure out which rank to send the vector
      const auto mb_offset = mb_i * num_local_cols * num_local_output_rows;
      const auto values_offest = mb_i * num_local_cols * num_local_values_rows;
      const auto ind_offset = mb_i * num_local_output_rows;

      for(size_t row = gidx; row < num_local_output_rows * warp_size; row += nthreadsx){
        const auto ind = static_cast<El::Int>(floor(indices[ind_offset + row / warp_size]));
        if (ind >= 0 ){ 
          const int pe = (pe_group * pe_stride) + (ind / num_local_values_rows);
          const int local_ind = ind % num_local_values_rows;

          nvshmemx_getmem_nbi_warp(&output[mb_offset + (row / warp_size) * num_local_cols],
                                  &shared_buffer[values_offest + local_ind * num_local_cols],
                                  num_local_cols * sizeof(DataType),
                                  pe);
        }
      }
    }
  }

  /**
   * @brief NVSHMEMScatter kernel
   * 
   */
  template <typename DataType>
  __global__ void Scatter_NVSHMEM_Kernel(
      const DataType* __restrict__ values,
      const DataType* __restrict__ indices,
      DataType* __restrict__ outputs,
      const int mini_batch_size,
      const int num_local_values_rows,
      const int num_cols,
      const int num_local_output_rows,
      const int pe_group,
      const int pe_stride){
    // Indices
    const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
    const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;
    const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;

    const size_t nthreadsx = gridDim.x * blockDim.x;
    const size_t nthreadsy = gridDim.y * blockDim.y;
    const size_t nthreadsz = gridDim.z * blockDim.z;


    for (size_t mb_i = gidz; mb_i < mini_batch_size; mb_i += nthreadsz){
      const auto values_offset = mb_i * num_local_values_rows * num_cols;
      const auto output_offset = mb_i * num_local_output_rows * num_cols;
      const auto indices_offset = mb_i * num_local_values_rows;

      for(size_t row = gidy; row < num_local_values_rows; row += nthreadsy){
        // Figure out which rank to send the vector
        const auto ind = static_cast<El::Int>(floor(indices[indices_offset + row]));
        if (ind >= 0){
          const int pe = (pe_group * pe_stride) + (ind / num_local_output_rows);
          const int local_ind = ind % num_local_output_rows;

          for(size_t i = gidx; i < num_cols; i += nthreadsx){
            const auto val = values[values_offset + row * num_cols + i];
            atomic_add(&outputs[output_offset + (local_ind  * num_cols) + i], val, pe);
          }
        }
      }
    }
  }
  } // namespace <anon>

  namespace tensor{
    
    template<typename DataType>
    void
    ScatterNVSHMEM<DataType>
    ::scatter(const DataType* values,
              const DataType* indices,
              DataType* output,
              const size_t local_mini_batch_size,
              const size_t values_rows_size,
              const size_t values_cols_size,
              const size_t output_rows_size){
                
      const auto buffer_size = local_mini_batch_size * output_rows_size * values_cols_size;
      const auto output_size = local_mini_batch_size * output_rows_size * values_cols_size;
      
      if (buffer_size == 0){ // No work to be done here
        nvshmemx_barrier_all_on_stream(m_stream);
        return ;
      }

      // Make sure the buffers are large enough
      ensure_buffer(buffer_size);
      DataType* output_buffer_ptr = static_cast<DataType*>(m_output_buffer.get()); 
      cudaMemsetAsync(output, 0, sizeof(DataType) * output_size, m_stream); // Not sure why this seems to be neccessary
      cudaStreamSynchronize(m_stream);
      // To Do: This would result in overcommitted allocation for 
      //        single mini-batch case. Add different thread configuration to 
      //        optimize for the case of single mini_batch - S.Z
      constexpr size_t block_size_x = 1; 
      constexpr size_t block_size_y = 16;  // full-warp
      dim3 block_dims, grid_dims;
      block_dims.x = block_size_x;
      block_dims.y = block_size_y;
      block_dims.z = 1;

      grid_dims.z = (local_mini_batch_size + block_dims.z -1) / block_dims.z;
      grid_dims.y = (values_rows_size + block_dims.y - 1) / block_dims.y;
      grid_dims.x = (values_cols_size + block_dims.x - 1) / block_dims.x;
      sync();
      cudaDeviceSynchronize();
      
      Scatter_NVSHMEM_Kernel<<<grid_dims, block_dims, 0, m_stream>>>(values,
                                                                     indices,
                                                                     output_buffer_ptr,
                                                                     local_mini_batch_size,
                                                                     values_rows_size,
                                                                     values_cols_size,
                                                                     output_rows_size,
                                                                     m_group,
                                                                     m_stride);
      cudaDeviceSynchronize();
      nvshmemx_quiet_on_stream(m_stream);
      sync();
      // Copy the local workspace buffer onto the output matrix 
      grid_dims.y = (output_rows_size + block_dims.y - 1) / block_dims.y;
      cudaMemcpyAsync(output,
                      output_buffer_ptr,
                      sizeof(DataType) * buffer_size,
                      cudaMemcpyDeviceToDevice,
                      m_stream);
      cudaStreamSynchronize(m_stream);
    }

    template<typename DataType>
    void
    GatherNVSHMEM<DataType>
    ::gather(const DataType* values,
             const DataType* indices,
             DataType* output,
             const size_t local_mini_batch_size,
             const size_t values_rows_size,
             const size_t values_cols_size,
             const size_t output_rows_size){

      const auto buffer_size = local_mini_batch_size * values_rows_size * values_cols_size;
      const auto output_size = local_mini_batch_size * output_rows_size * values_cols_size;
      // The NVSHMEM workspace buffer is the size of the local output matrix 
      if (buffer_size == 0){ // No work to be done here
        nvshmemx_barrier_all_on_stream(m_stream);
        return ;
      }
      ensure_buffer(buffer_size);

      cudaMemset(output, 0, sizeof(DataType) * output_size); // Not sure why this seems to be neccessary

      constexpr size_t block_size_x = 32;
      constexpr size_t block_size_y = 16;

      // Copy the local values matrix onto the workspace buffer 
      // The values matrix must be in symmetric heap in order to be remotely accessible 
      // Change grid dimensions for copy kernel. Will be removed in the furure - S.Z

      dim3 block_dims, grid_dims;
      block_dims.x = block_size_x;
      block_dims.y = block_size_y;
      block_dims.z = 1;

      grid_dims.x = (values_cols_size + block_dims.x - 1) / block_dims.x;
      grid_dims.y = (values_rows_size + block_dims.y - 1) / block_dims.y;
      grid_dims.z = (local_mini_batch_size + block_dims.z - 1) / block_dims.z;
          
      cudaMemcpyAsync(static_cast<DataType*>(m_output_buffer.get()),
                      values,
                      sizeof(DataType) * buffer_size,
                      cudaMemcpyDeviceToDevice,
                      m_stream);
      cudaStreamSynchronize(m_stream);
      sync();  // Barrier to ensure all PEs are available

      // To Do: This would result in overcommitted allocation for 
      //        single mini-batch case. Add different thread configuration to 
      //        optimize for the case of single mini_batch - S.Z

      block_dims.x = block_size_x;
      block_dims.y = 1;
      block_dims.z = 1;

      grid_dims.x = (output_rows_size + block_dims.x - 1) / block_dims.x;
      grid_dims.y = (local_mini_batch_size + block_dims.y - 1)/ block_dims.y;
      grid_dims.z = 1;

      Gather_NVSHMEM_Kernel<<<grid_dims, block_dims, 0, m_stream>>>(static_cast<DataType*>(m_output_buffer.get()),
                                                                    indices,
                                                                    output,
                                                                    local_mini_batch_size,
                                                                    values_rows_size,
                                                                    values_cols_size,
                                                                    output_rows_size,
                                                                    m_group,
                                                                    m_stride);
      nvshmemx_quiet_on_stream(m_stream);
      sync();
    }

  
  #define SCATTER_ETI(T)                            \
    template void ScatterNVSHMEM<T>::scatter(       \
      const T* values,                              \
      const T* indices,                             \
      T * output,                                   \
      const size_t local_mini_batch_size,           \
      const size_t values_rows_size,                \
      const size_t values_cols_size,                \
      const size_t output_rows_size);                    

  SCATTER_ETI(float)  
  SCATTER_ETI(double)
  #undef SCATTER_ETI

  #define GATHER_ETI(T)                            \
    template void GatherNVSHMEM<T>::gather(        \
      const T* values,                             \
      const T* indices,                            \
      T * output,                                  \
      const size_t local_mini_batch_size,          \
      const size_t values_rows_size,               \
      const size_t values_cols_size,               \
      const size_t output_rows_size);                    

  GATHER_ETI(float)  
  GATHER_ETI(double)
  #undef GATHER_ETI 

  } // namespace <distconv::tensor>
} //  namespace <distconv>
#endif 
