#include "gpu_utils.hpp"

#include "errors.hpp"
#include "logging.hpp"

bool lbannv2::gpu::is_integrated() noexcept
{
#if LBANNV2_WITH_MI300A
  return true;
#else
#if LBANNV2_HAS_ROCM
  hipDeviceProp_t props;
  if (hipGetDeviceProperties(&props, current_device()) == hipSuccess)
    return props.integrated;
  LBANNV2_ERROR("Failed to get device properties of current HIP device {}.",
                current_device());
#endif
#endif
  return false;
}

c10::DeviceIndex lbannv2::gpu::num_devices() noexcept
{
#if LBANNV2_HAS_GPU
  return c10_gpu::device_count();
#else
  return 0;
#endif
}

c10::DeviceIndex lbannv2::gpu::current_device()
{
#if LBANNV2_HAS_GPU
  return c10_gpu::current_device();
#else
  return -1;
#endif
}

void lbannv2::gpu::set_device(c10::DeviceIndex const d)
{
  LBANNV2_TRACE("lbannv2::gpu::set_device(d={})", d);
  LBANNV2_ASSERT_ALWAYS(d >= 0 && d < num_devices());
#if LBANNV2_HAS_GPU
  c10_gpu::set_device(d, false);
#endif
}

#if LBANNV2_HAS_GPU
#if LBANNV2_HAS_CUDA
#define lbannv2StreamCreate cudaStreamCreate
#define lbannv2StreamCreateWithFlags cudaStreamCreateWithFlags
#define lbannv2StreamNonBlocking cudaStreamNonBlocking
#define lbannv2StreamSync cudaStreamSynchronize
#define lbannv2StreamDestroy cudaStreamDestroy
#elif LBANNV2_HAS_ROCM
#define lbannv2StreamCreate hipStreamCreate
#define lbannv2StreamCreateWithFlags hipStreamCreateWithFlags
#define lbannv2StreamNonBlocking hipStreamNonBlocking
#define lbannv2StreamSync hipStreamSynchronize
#define lbannv2StreamDestroy hipStreamDestroy
#endif

auto lbannv2::gpu::make_stream() -> Stream_t
{
  Stream_t stream;
  LBANNV2_CHECK_GPU(lbannv2StreamCreate(&stream));
  LBANNV2_TRACE("lbannv2::gpu::make_stream(): created stream {}",
                (void*) stream);
  return stream;
}

auto lbannv2::gpu::make_nonblocking_stream() -> Stream_t
{
  Stream_t stream;
  LBANNV2_CHECK_GPU(
    lbannv2StreamCreateWithFlags(&stream, lbannv2StreamNonBlocking));
  LBANNV2_TRACE("lbannv2::gpu::make_nonblocking_stream(): created stream {}",
                (void*) stream);
  return stream;
}

void lbannv2::gpu::sync(Stream_t const stream)
{
  LBANNV2_CHECK_GPU(lbannv2StreamSync(stream));
  LBANNV2_TRACE("lbannv2::gpu::sync(stream={})", (void const*) stream);
}

void lbannv2::gpu::destroy_stream(Stream_t const stream)
{
  LBANNV2_CHECK_GPU(lbannv2StreamDestroy(stream));
  LBANNV2_TRACE("lbannv2::gpu::destroy_stream(stream={})", (void*) stream);
}

#endif
