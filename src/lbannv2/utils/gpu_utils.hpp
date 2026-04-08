#pragma once
#include <lbannv2_config.h>

#include <c10/core/Device.h>

#if LBANNV2_HAS_CUDA

#include <lbannv2/utils/logging.hpp>

#include <c10/cuda/CUDAFunctions.h>

#include <stdexcept>

#include <cuda_runtime.h>

#define LBANNV2_CHECK_GPU(cmd)                                                 \
  do                                                                           \
  {                                                                            \
    auto const lbannv2_check_gpu_status = (cmd);                               \
    if (lbannv2_check_gpu_status != cudaSuccess)                               \
    {                                                                          \
      LBANNV2_DEBUG("CUDA command \"" #cmd "\" failed. Error: {}",             \
                    cudaGetErrorString(lbannv2_check_gpu_status));             \
      throw std::runtime_error("CUDA command \"" #cmd "\" failed.");           \
    }                                                                          \
  } while (0)

#elif LBANNV2_HAS_ROCM

#include <lbannv2/utils/logging.hpp>

#include <c10/hip/HIPFunctions.h>
#include <c10/hip/HIPStream.h>

#include <stdexcept>

#include <hip/hip_runtime.h>

#define LBANNV2_CHECK_GPU(cmd)                                                 \
  do                                                                           \
  {                                                                            \
    auto const lbannv2_check_gpu_status = (cmd);                               \
    if (lbannv2_check_gpu_status != hipSuccess)                                \
    {                                                                          \
      LBANNV2_DEBUG("HIP command \"" #cmd "\" failed. Error: {}",              \
                    hipGetErrorString(lbannv2_check_gpu_status));              \
      throw std::runtime_error("HIP command \"" #cmd "\" failed.");            \
    }                                                                          \
  } while (0)
#endif

namespace lbannv2
{
#if LBANNV2_USE_C10_HIP_NAMESPACE_AND_SYMBOLS
namespace c10_gpu = c10::hip;
using TorchGPUStream_t = c10::hip::HIPStream;
inline auto& getDeviceCurrentStream = c10::hip::getCurrentHIPStream;
#elif LBANNV2_HAS_GPU
namespace c10_gpu = c10::cuda;
using TorchGPUStream_t = c10::cuda::CUDAStream;
inline auto& getDeviceCurrentStream = c10::cuda::getCurrentCUDAStream;
#endif

inline constexpr bool has_cuda() noexcept
{
  return LBANNV2_HAS_CUDA;
}

inline constexpr bool has_hip() noexcept
{
  return LBANNV2_HAS_ROCM;
}

inline constexpr bool has_gpu() noexcept
{
  return LBANNV2_HAS_GPU;
}

namespace gpu
{

#if LBANNV2_HAS_CUDA
using Stream_t = cudaStream_t;
#elif LBANNV2_HAS_ROCM
using Stream_t = hipStream_t;
#endif

// Returns 'false' if no GPU support
bool is_integrated() noexcept;

// Returns 0 if no GPU support
c10::DeviceIndex num_devices() noexcept;

// Returns -1 if no GPU support
c10::DeviceIndex current_device();

// Throws if d >= num_devices() or d < 0.
void set_device(c10::DeviceIndex d);

#if LBANNV2_HAS_GPU
Stream_t make_stream();
Stream_t make_nonblocking_stream();
void sync(Stream_t);
void destroy_stream(Stream_t);
#endif

}  // namespace gpu

}  // namespace lbannv2
