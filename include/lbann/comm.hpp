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

#ifndef LBANN_COMM_HPP_INCLUDED
#define LBANN_COMM_HPP_INCLUDED

#include "base.hpp"

#ifdef LBANN_HAS_CUDA
#include <cuda_runtime.h>
#endif // LBANN_HAS_CUDA
#ifdef LBANN_HAS_ALUMINUM
#include <Al.hpp>
#endif // LBANN_HAS_ALUMINUM

#include "detect_El_mpi.hpp"

#include <map>
#include <typeindex>
#include <vector>

namespace lbann {

#ifdef LBANN_HAS_ALUMINUM
/** Convert an MPI_Op to an Aluminum reduction operator. */
::Al::ReductionOperator mpi_op_to_al_op(El::mpi::Op op);
#endif

namespace Al {

/** Dummy Aluminum backend. */
class dummy_backend
{
public:
  using req_type = int;
  static constexpr req_type null_req = 0;
};

// Define aliases for Aluminum backends
#ifdef LBANN_HAS_ALUMINUM
using mpi_backend = ::Al::MPIBackend;
#else
using mpi_backend = lbann::Al::dummy_backend;
#endif // LBANN_HAS_ALUMINUM
using mpi_req_type = mpi_backend::req_type;
static const mpi_req_type mpi_null_req = mpi_backend::null_req;
/// @todo MPI-CUDA backend
#if defined(LBANN_HAS_ALUMINUM) && defined(AL_HAS_NCCL)
using nccl_backend = ::Al::NCCLBackend;
// LBANN does its own synchronization on this.
#else
using nccl_backend = lbann::Al::dummy_backend;
#endif // defined(LBANN_HAS_ALUMINUM) && defined(AL_HAS_NCCL)
using nccl_req_type = nccl_backend::req_type;
static const nccl_req_type nccl_null_req = nccl_backend::null_req;
#if defined(LBANN_HAS_ALUMINUM) && defined(AL_HAS_MPI_CUDA)
using mpicuda_backend = ::Al::MPICUDABackend;
#else
using mpicuda_backend = lbann::Al::dummy_backend;
#endif // defined(LBANN_HAS_ALUMINUM) && defined(AL_HAS_MPI_CUDA)
#if defined(LBANN_HAS_ALUMINUM) && defined(AL_HAS_HOST_TRANSFER)
using hosttransfer_backend = ::Al::HostTransferBackend;
#else
using hosttransfer_backend = lbann::Al::dummy_backend;
#endif // defined(LBANN_HAS_ALUMINUM) && defined(AL_HAS_HOST_TRANSFER)
using mpicuda_req_type = mpicuda_backend::req_type;
static const mpicuda_req_type mpicuda_null_req = mpicuda_backend::null_req;

/** Wrapper for Aluminum non-blocking routine requests. */
struct request
{
  mpi_req_type mpi_req = mpi_null_req;
  nccl_req_type nccl_req = nccl_null_req;
  mpicuda_req_type mpicuda_req = mpicuda_null_req;
};

} // namespace Al

/* Notes on Synchronization
 *
 * The updated interface exposes a synchronization handle/device
 * tagging mechanism used by Hydrogen: El::SyncInfo<D>, where D is an
 * El::Device. When operating on Matrix objects, this should be
 * handled automagically, assuming the Matrix is setup properly. Users
 * must be aware of this when making MPI calls through Hydrogen or
 * through lbann_comm with raw data buffers (T[]).
 *
 * When dealing with El::Matrix objects, users should be aware of the
 * following. There is no synchronization for CPU objects
 * (El::SyncInfo<El::Device::CPU> is an empty struct), but GPU Matrix
 * objects now have an associated stream and event. These are
 * GPUManager::Stream() and GPUManager::Event() by default, resp., but
 * can be overriden by a user. Note: the Matrix never owns these; it
 * will not free these resources at destruction. There are many
 * methods in which multiple El::Matrix objects might interact. This
 * should work properly; otherwise, report bugs to benson31.
 *
 * When dealing with raw data (T[]), users should be aware of the
 * following. In the near future, all El::mpi functions will have an
 * El::SyncInfo object as their last parameter, and it will be a
 * required parameter. In lbann_comm, this means that when the call
 * trickles down to an El::mpi function, an appropriate El::SyncInfo
 * must be available. Since many of LBANN's uses of this interface are
 * for communicating CPU buffers, there is "shortcut" API that assumes
 * the data is CPU memory, thus providing the default
 * El::SyncInfo<El::Device::CPU> object to El::mpi. If a user wishes
 * to communicate GPU data, they must use the "full" API, which adds a
 * final El::SyncInfo parameter to the function. This ensures the
 * appropriate synchronization semantics, especially when working with
 * Aluminum as the communication frontend.
 */

/**
 * Manage communication.
 * This supports separate trainers, each of which are split over potentially
 * several processes. Every trainer is split over the same number of processes.
 * The corresponding processes between trainers are on the "inter-trainer
 * communicator".
 * You can also do point-to-point or broadcast communication to arbitrary sets
 * of processes.
 */
class lbann_comm
{
public:
  /**
   * Init communicators for trainers each with procs_per_trainer processes,
   * defaulting to every process in one trainer.
   */
  lbann_comm(int procs_per_trainer = 0,
             El::mpi::Comm world = El::mpi::COMM_WORLD.GetMPIComm());
  /** Don't allow copying; it doesn't make sense for the communicator. */
  lbann_comm(const lbann_comm&) = delete;
  /** Don't allow assignment; it doesn't make sense for the communicator. */
  lbann_comm& operator=(const lbann_comm&) = delete;
  ~lbann_comm();

  /**
   * Split communicators so each trainer has procs_per_trainer processes.
   * If you call this multiple times, it will invalidate existing grids
   * and communicators.
   */
  void split_trainers(int procs_per_trainer);

  /** Get which trainer this process is in. */
  inline int get_trainer_rank() const { return trainer_rank; }
  /** Get the rank of this process in its trainer. */
  inline int get_rank_in_trainer() const { return rank_in_trainer; }
  /** Get my rank in COMM_WORLD. */
  inline int get_rank_in_world() const
  {
    return El::mpi::Rank(get_world_comm());
  }
  /** Return the COMM_WORLD rank of the rank'th processor in trainer. */
  inline int get_world_rank(int trainer, int rank) const
  {
    return procs_per_trainer * trainer + rank;
  }
  /** Return the "rank" of the trainer that this rank is in */
  inline int map_world_rank_to_trainer_rank(int world_rank) const
  {
    return (world_rank / procs_per_trainer);
  }
  /** Return the "rank" within the trainer that this rank is in */
  inline int map_world_rank_to_rank_in_trainer(int world_rank) const
  {
    return (world_rank % procs_per_trainer);
  }
  /** Return the rank of the master process in this trainer. */
  inline int get_trainer_master() const { return 0; }
  /** Return the rank of the inter-trainer master process. */
  inline int get_intertrainer_master() const { return 0; }
  /** Return the rank of the world master process. */
  inline int get_world_master() const { return 0; }
  /** Return true if this process is the master process in its trainer. */
  inline bool am_trainer_master() const
  {
    return get_rank_in_trainer() == get_trainer_master();
  }
  /** Return true if this process is the world master process. */
  inline bool am_world_master() const
  {
    return get_rank_in_world() == get_world_master();
  }
  /** Return a grid to use for this trainer. */
  inline El::Grid& get_trainer_grid() { return *grid; }
  /** Return a read-only grid to use for this trainer. */
  inline const El::Grid& get_trainer_grid() const { return *grid; }
  /** Return the total number of trainers. */
  inline int get_num_trainers() const { return num_trainers; }
  /* Return the number of processes in a trainer. */
  inline int get_procs_per_trainer() const { return procs_per_trainer; }
  /** Return the number of processes in a compute node. */
  inline int get_procs_per_node() const { return procs_per_node; }
  /** Return the total number of ranks. */
  inline int get_procs_in_world() const
  {
    return El::mpi::Size(get_world_comm());
  }
  /** Return the rank of this process within its compute node. */
  inline int get_rank_in_node() const { return rank_in_node; }
  /** Return true if rank (in COMM_WORLD) is on this compute node. */
  inline bool is_world_rank_on_node(int rank) const
  {
    return std::find(world_ranks_on_node.begin(),
                     world_ranks_on_node.end(),
                     rank) != world_ranks_on_node.end();
  }

  /** Get default number of threads per process.
   *  This is the number of OpenMP threads to use for parallel
   *  regions, provided omp_set_num_threads has not been called or the
   *  num_threads directive has not been provided.
   */
  inline int get_default_threads_per_proc() const { return threads_per_proc; }

  /** Reset the number of threads per process to the default. */
  void reset_threads();

  /** Perform a sum reduction of mat over the inter-trainer communicator. */
  void intertrainer_sum_matrix(AbsMat& mat);
  void intertrainer_sum_matrix(AbsDistMat& mat);
  /** Broadcast mat over the inter-trainer communicator starting from root. */
  void intertrainer_broadcast_matrix(AbsMat& mat, int root);
  void intertrainer_broadcast_matrix(AbsDistMat& mat, int root);

  /// Broadcast a scalar value over an arbitrary communicator
  template <typename T, bool S = is_instantiated_El_mpi_type<T>::value>
  void broadcast(int root, T& val, const El::mpi::Comm& c);

  template <typename T>
  void broadcast_custom(int root, T& val, const El::mpi::Comm& c) const;
  template <typename T>
  void broadcast_native(int root, T& val, const El::mpi::Comm& c) const;

  /// World broadcast of a scalar.
  template <typename T> void world_broadcast(int root, T& val);
  /// Inter-trainer broadcast of a scalar.
  template <typename T> void intertrainer_broadcast(int root, T& val);
  /// Within-trainer broadcast of a scalar.
  template <typename T> void trainer_broadcast(int root, T& val);

  /**
   * Broadcast a buffer over an arbitrary communicator assuming that
   * the buffer space is already allocated.
   */

  // Default to cpu memory
  template <typename T>
  void
  broadcast(const int root, T* data, const int count, const El::mpi::Comm& c);

  template <typename T,
            El::Device D,
            bool S = is_instantiated_El_mpi_type<T>::value>
  void broadcast(const int root,
                 T* data,
                 const int count,
                 const El::mpi::Comm& c,
                 El::SyncInfo<D> const& syncInfo);

  /// World broadcast of a buffer.
  template <typename T>
  void world_broadcast(const int root, T* data, const int count);

  template <typename T, El::Device D>
  void world_broadcast(const int root,
                       T* data,
                       const int count,
                       El::SyncInfo<D> const& syncInfo);
  /// Inter-trainer broadcast of a buffer.
  template <typename T>
  void intertrainer_broadcast(const int root, T* data, const int count);
  template <typename T, El::Device D>
  void intertrainer_broadcast(const int root,
                              T* data,
                              const int count,
                              El::SyncInfo<D> const& syncInfo);
  /// Within-trainer broadcast of a buffer.
  template <typename T>
  void trainer_broadcast(const int root, T* data, const int count);

  template <typename T, El::Device D>
  void trainer_broadcast(const int root,
                         T* data,
                         const int count,
                         El::SyncInfo<D> const& syncInfo);

  /**
   * Resize vector<> over an arbitrary communicator to match the one on root.
   */
  template <typename T>
  size_t resize(const int root, std::vector<T>& data, const El::mpi::Comm& c);

  /**
   * Broadcast vector<> over an arbitrary communicator;
   * vector<> for non-root processes will be resized as needed.
   */
  template <typename T>
  void broadcast(const int root, std::vector<T>& data, const El::mpi::Comm& c);
  /// Broadcast vector<> to world.
  template <typename T> void world_broadcast(int root, std::vector<T>& data);
  /**
   * Broadcast vector<> within trainer;
   * vector<> for non-root processes will be resized as needed.
   */
  /// Broadcast vector<> across trainers.
  template <typename T>
  void intertrainer_broadcast(int root, std::vector<T>& data);
  /// Broadcast vector<> within trainer.
  template <typename T> void trainer_broadcast(int root, std::vector<T>& data);

  /**
   * Keep track of the number of broadcast bytes transmitted and received
   */
  void count_bytes_broadcast(const size_t bytes, const int rank, const int root)
  {
    if (rank == root) {
      bytes_sent += bytes;
    }
    else {
      bytes_received += bytes;
    }
  }

  /** Allgather over an arbitrary communicator */
  template <typename T>
  void all_gather(const T* src,
                  int src_count,
                  T* rcv,
                  int rcv_count,
                  const El::mpi::Comm& c);
  template <typename T, El::Device D>
  void all_gather(const T* src,
                  int src_count,
                  T* rcv,
                  int rcv_count,
                  const El::mpi::Comm& c,
                  El::SyncInfo<D> const& syncInfo);

  /**
   * Allgatherv over an arbitrary communicator;
   * all vectors must be correctly sized prior to entry.
   */
  template <typename T>
  void all_gather(std::vector<T>& src,
                  std::vector<T>& rcs,
                  std::vector<int>& rcv_counts,
                  std::vector<int>& rcv_disp,
                  const El::mpi::Comm& c);
  /**
   * Allgatherv over a trainer communicator;
   * all vectors must be correctly sized prior to entry.
   */
  template <typename T>
  void trainer_all_gather(std::vector<T>& src,
                          std::vector<T>& rcs,
                          std::vector<int>& rcv_counts,
                          std::vector<int>& rcv_disp);
  /**
   * Allgather for a single element over an arbitrary communicator;
   * std::vector<T> &data must be correctly sized prior to entry.
   */
  template <typename T>
  void all_gather(T& src, std::vector<T>& data, const El::mpi::Comm& c);
  /**
   * Allgather for a single element over the world communicator;
   * std::vector<T> &data must be correctly sized prior to entry.
   */
  template <typename T> void world_all_gather(T& src, std::vector<T>& data);
  /**
   * Allgather for a single element over the trainer communicator;
   * std::vector<T> &data must be correctly sized prior to entry.
   */
  template <typename T> void trainer_all_gather(T& src, std::vector<T>& data);

  /** Within-trainer scalar gather (for non-root processes). */
  template <typename T> void trainer_gather(T snd, int root);
  /** Within-trainer scalar gather (for root processes). */
  template <typename T> void trainer_gather(T snd, T* rcv);
  /** Within-trainer scalar-array gather (for non-root processes). */
  template <typename T> void trainer_gather(T* snd, int count, int root);
  /** Within-trainer scalar-array gather (for root processes). */
  template <typename T> void trainer_gather(T* snd, int count, T* rcv);
  /** Within-trainer variable-length-array gather (for non-root processes). */
  template <typename T> void trainer_gatherv(T* snd, int count, int root);
  template <typename T>
  void trainer_gatherv(T* snd,
                       int count,
                       T* rcv,
                       int* rcv_counts,
                       int* rcv_displacements);
  /** Inter-trainer gather (for non-root processes). */
  template <typename T> void intertrainer_gather(T snd, int root);
  /** Inter-trainer gather (for root processes). */
  template <typename T> void intertrainer_gather(T snd, std::vector<T>& rcv);
  /** Inter-trainer scalar-array gather (for non-root processes). */
  template <typename T> void intertrainer_gather(T* snd, int count, int root);
  /** Inter-trainer scalar-array gather (for root processes). */
  template <typename T> void intertrainer_gather(T* snd, int count, T* rcv);
  /** Scalar gather (for non-root processes). */
  template <typename T> void gather(T snd, int root, const El::mpi::Comm& c);
  /** Scalar gather (for root processes). */
  template <typename T> void gather(T snd, T* rcv, const El::mpi::Comm& c);
  /** Scalar gather (for root processes). */
  template <typename T>
  void gather(T snd, std::vector<T>& rcv, const El::mpi::Comm& c);
  /** Scalar-array gather (for non-root processes). */
  template <typename T>
  void gather(T* snd, int count, int root, const El::mpi::Comm& c);
  template <typename T, El::Device D>
  void gather(T* snd,
              int count,
              int root,
              const El::mpi::Comm& c,
              El::SyncInfo<D> const& syncInfo);
  /** Scalar-array gather (for root processes). */
  template <typename T>
  void gather(T* snd, int count, T* rcv, const El::mpi::Comm& c);
  template <typename T, El::Device D>
  void gather(T* snd,
              int count,
              T* rcv,
              const El::mpi::Comm& c,
              El::SyncInfo<D> const& syncInfo);
  /** Scalar scatter (for non-root processes). */
  template <typename T> T scatter(int root, const El::mpi::Comm& c);
  /** Scalar scatter (for root processes). */
  template <typename T> T scatter(T* snd, const El::mpi::Comm& c);
  /** Inter-trainer reduce (for non-root processes). */
  template <typename T>
  void intertrainer_reduce(T snd, int root, El::mpi::Op op = El::mpi::SUM);
  /** Inter-trainer reduce (for root processes). */
  template <typename T>
  T intertrainer_reduce(T snd, El::mpi::Op op = El::mpi::SUM);
  /** Within-trainer reduce (for non-root processes). */
  template <typename T>
  void trainer_reduce(T snd, int root, El::mpi::Op op = El::mpi::SUM);
  /** Within-trainer reduce (for root processes). */
  template <typename T> T trainer_reduce(T snd, El::mpi::Op op = El::mpi::SUM);
  /** Within-trainer scalar array reduce (for non-root processes). */
  template <typename T>
  void
  trainer_reduce(T* snd, int count, int root, El::mpi::Op op = El::mpi::SUM);
  /** Within-trainer scalar array reduce (for root processes). */
  template <typename T>
  void trainer_reduce(T* snd, int count, T* rcv, El::mpi::Op op = El::mpi::SUM);
  /** Scalar reduce (for non-root processes). */
  template <typename T>
  void reduce(T snd,
              int root,
              const El::mpi::Comm& c,
              El::mpi::Op op = El::mpi::SUM);
  /** Scalar reduce (for root processes). */
  template <typename T>
  T reduce(T snd, const El::mpi::Comm& c, El::mpi::Op op = El::mpi::SUM);

  /** Scalar-array reduce (for non-root processes). */
  // Op is "SUM"
  template <typename T>
  void reduce(T* snd, int count, int root, const El::mpi::Comm& c);
  template <typename T, El::Device D>
  void reduce(T* snd,
              int count,
              int root,
              const El::mpi::Comm& c,
              El::SyncInfo<D> const& syncInfo);

  template <typename T>
  void
  reduce(T* snd, int count, int root, const El::mpi::Comm& c, El::mpi::Op op);
  template <typename T, El::Device D>
  void reduce(T* snd,
              int count,
              int root,
              const El::mpi::Comm& c,
              El::mpi::Op op,
              El::SyncInfo<D> const& syncInfo);
  /** Scalar-array reduce (for root processes). */
  template <typename T, El::Device D>
  void reduce(T* snd,
              int count,
              T* rcv,
              const El::mpi::Comm& c,
              El::SyncInfo<D> const& syncInfo);
  template <typename T>
  void reduce(T* snd, int count, T* rcv, const El::mpi::Comm& c);

  template <typename T>
  void
  reduce(T* snd, int count, T* rcv, const El::mpi::Comm& c, El::mpi::Op op);
  template <typename T, El::Device D>
  void reduce(T* snd,
              int count,
              T* rcv,
              const El::mpi::Comm& c,
              El::mpi::Op op,
              El::SyncInfo<D> const& syncInfo);
  /** Inter-trainer all-reduce. */
  template <typename T>
  T intertrainer_allreduce(T snd, El::mpi::Op op = El::mpi::SUM);
  /** Within-trainer all-reduce. */
  template <typename T>
  T trainer_allreduce(T snd, El::mpi::Op op = El::mpi::SUM);
  /** Scalar array within-trainer all-reduce. */
  template <typename T>
  void
  trainer_allreduce(T* snd, int count, T* rcv, El::mpi::Op op = El::mpi::SUM);
  /** Scalar allreduce. */
  template <typename T>
  T allreduce(T snd, const El::mpi::Comm& c, El::mpi::Op op = El::mpi::SUM);

  // FIXME (trb): Based on the backend choice of "MPIBackend", I'm
  // assuming this is intended as a CPU-only call.
  /** Scalar-array allreduce. */
  template <typename T>
  void allreduce(T* snd,
                 int count,
                 T* rcv,
                 const El::mpi::Comm& c,
                 El::mpi::Op op = El::mpi::SUM);
  /** In-place scalar-array allreduce. */
  template <typename T>
  void allreduce(T* data,
                 int count,
                 const El::mpi::Comm& c,
                 El::mpi::Op op = El::mpi::SUM);
  /** Matrix allreduce. */
  template <typename TensorDataType>
  void allreduce(El::AbstractMatrix<TensorDataType>& m,
                 const El::mpi::Comm& c,
                 El::mpi::Op op = El::mpi::SUM);
  /** Matrix allreduce. */
  template <typename TensorDataType>
  void allreduce(El::AbstractDistMatrix<TensorDataType>& m,
                 const El::mpi::Comm& c,
                 El::mpi::Op op = El::mpi::SUM);
  /** Non-blocking matrix allreduce.
   *  If LBANN has not been built with Aluminum, then this calls a
   *  blocking matrix allreduce.
   */
  template <typename TensorDataType>
  void nb_allreduce(El::AbstractMatrix<TensorDataType>& m,
                    const El::mpi::Comm& c,
                    Al::request& req,
                    El::mpi::Op op = El::mpi::SUM);
  /** Non-blocking matrix allreduce.
   *  If LBANN has not been built with Aluminum, then this calls a
   *  blocking matrix allreduce.
   */
  template <typename TensorDataType>
  void nb_allreduce(El::AbstractDistMatrix<TensorDataType>& m,
                    const El::mpi::Comm& c,
                    Al::request& req,
                    El::mpi::Op op = El::mpi::SUM);
  /** Non-blocking in-place scalar-array allreduce.
   *  If LBANN has not been built with Aluminum, then this calls a blocking
   *  allreduce.
   *  This currently only supports host pointers (i.e. the MPI backend).
   */
  template <typename T>
  void nb_allreduce(T* data,
                    int count,
                    const El::mpi::Comm& c,
                    Al::request& req,
                    El::mpi::Op op = El::mpi::SUM);

  /** Wait for a all non-blocking requests to complete. */
  template <typename T> void wait_all(std::vector<El::mpi::Request<T>>& req);

  /** Wait for a non-blocking request to complete. */
  template <typename T> void wait(El::mpi::Request<T>& req);

  /** Wait for a non-blocking request to complete. */
  void wait(Al::request& req);
  /** Test whether a non-blocking request has completed; true if it has. */
  bool test(Al::request& req);

  /** Barrier among the inter-trainer processes. */
  void intertrainer_barrier();
  /** Barrier among processes in this trainer. */
  void trainer_barrier();
  /** Barrier among all processes. */
  void global_barrier();
  /** Barrier on an arbitrary communicator. */
  void barrier(const El::mpi::Comm& c);

  /** Send a buffer to rank in trainer. */
  template <typename T>
  void send(const T* data, int count, int trainer, int rank);
  template <typename T, El::Device D>
  void send(const T* data,
            int count,
            int trainer,
            int rank,
            El::SyncInfo<D> const& syncInfo);
  template <typename T, El::Device D>
  void
  send(const T* data, int count, int trainer, El::SyncInfo<D> const& syncInfo);
  void send(const AbsMat& mat, int trainer, int rank);
  void send(const DistMat& mat, int trainer, int rank);
  void send(const AbsMat& mat, int trainer)
  {
    send(mat, trainer, rank_in_trainer);
  }
  void send(const DistMat& mat, int trainer)
  {
    send(mat, trainer, rank_in_trainer);
  }

  /** Corresponding non-blocking sends. */
  template <typename T>
  void nb_send(const T* data,
               int count,
               int trainer,
               int rank,
               El::mpi::Request<T>& req);
  template <typename T>
  void nb_tagged_send(const T* data,
                      int count,
                      int rank,
                      int tag,
                      El::mpi::Request<T>& req,
                      const El::mpi::Comm& c);
  template <typename T>
  void nb_send(const T* data, int count, int trainer, El::mpi::Request<T>& req);
  void nb_send(const AbsMat& mat,
               int trainer,
               int rank,
               El::mpi::Request<DataType>& req);
  void nb_send(const DistMat& mat,
               int trainer,
               int rank,
               El::mpi::Request<DataType>& req);
  void nb_send(const AbsMat& mat, int trainer, El::mpi::Request<DataType>& req)
  {
    nb_send(mat, trainer, rank_in_trainer, req);
  }
  void nb_send(const DistMat& mat, int trainer, El::mpi::Request<DataType>& req)
  {
    nb_send(mat, trainer, rank_in_trainer, req);
  }

  /** Corresponding receive to send. */
  template <typename T> void recv(T* data, int count, int trainer, int rank);
  template <typename T> void recv(T* data, int count, int trainer);
  template <typename T> void recv(T* data, int count);
  template <typename T, El::Device D>
  void recv(T* data,
            int count,
            int trainer,
            int rank,
            El::SyncInfo<D> const& syncInfo);
  template <typename T, El::Device D>
  void recv(T* data, int count, int trainer, El::SyncInfo<D> const& syncInfo);
  void recv(AbsMat& mat, int trainer, int rank);
  void recv(DistMat& mat, int trainer, int rank);
  void recv(AbsMat& mat, int trainer) { recv(mat, trainer, rank_in_trainer); }
  void recv(DistMat& mat, int trainer) { recv(mat, trainer, rank_in_trainer); }
  /** As above, but receive from anyone. */
  template <typename T, El::Device D>
  void recv(T* data, int count, El::SyncInfo<D> const& syncInfo);
  void recv(AbsMat& mat);
  void recv(DistMat& mat);

  /** Corresponding non-blocking receives. */
  template <typename T>
  void
  nb_recv(T* data, int count, int trainer, int rank, El::mpi::Request<T>& req);
  template <typename T>
  void nb_tagged_recv(T* data,
                      int count,
                      int rank,
                      int tag,
                      El::mpi::Request<T>& req,
                      const El::mpi::Comm& c);

  template <typename T>
  void nb_recv(T* data, int count, int trainer, El::mpi::Request<T>& req);
  void
  nb_recv(AbsMat& mat, int trainer, int rank, El::mpi::Request<DataType>& req);
  void
  nb_recv(DistMat& mat, int trainer, int rank, El::mpi::Request<DataType>& req);
  void nb_recv(AbsMat& mat, int trainer, El::mpi::Request<DataType>& req)
  {
    nb_recv(mat, trainer, rank_in_trainer, req);
  }
  void nb_recv(DistMat& mat, int trainer, El::mpi::Request<DataType>& req)
  {
    nb_recv(mat, trainer, rank_in_trainer, req);
  }
  template <typename T>
  void nb_recv(T* data, int count, El::mpi::Request<T>& req);
  void nb_recv(AbsMat& mat, El::mpi::Request<DataType>& req);
  void nb_recv(DistMat& mat, El::mpi::Request<DataType>& req);

  /** Send/recv to/from ranks. */
  template <typename T, El::Device D>
  void sendrecv(const T* snd,
                int send_count,
                int send_trainer,
                int send_rank,
                T* rcv,
                int recv_count,
                int recv_trainer,
                int recv_rank);
  template <typename T, El::Device D>
  void sendrecv(const T* snd,
                int send_count,
                int send_trainer,
                T* rcv,
                int recv_count,
                int recv_trainer);

  template <typename T, El::Device D>
  void sendrecv(const T* snd,
                int send_count,
                int send_trainer,
                int send_rank,
                T* rcv,
                int recv_count,
                int recv_trainer,
                int recv_rank,
                El::SyncInfo<D> const& syncInfo);
  template <typename T, El::Device D>
  void sendrecv(const T* snd,
                int send_count,
                int send_trainer,
                T* rcv,
                int recv_count,
                int recv_trainer,
                El::SyncInfo<D> const& syncInfo);

  /** Determine the size (count) of an incoming message. */
  template <typename T> int get_count(int trainer, int rank);
  template <typename T> int get_count(int trainer);

  // Statistics methods.
  /** Return the number of trainer barriers performed. */
  inline size_t get_num_trainer_barriers() const
  {
    return num_trainer_barriers;
  }
  /** Return the number of inter-trainer barriers performed. */
  inline size_t get_num_intertrainer_barriers() const
  {
    return num_intertrainer_barriers;
  }
  /** Return the number of global barriers performed. */
  inline size_t get_num_global_barriers() const { return num_global_barriers; }
  /** Return the number of bytes sent. */
  inline size_t get_bytes_sent() const { return bytes_sent; }
  /** Return the number of bytes received. */
  inline size_t get_bytes_received() const { return bytes_received; }

  inline void reset_stats_counters()
  {
    num_trainer_barriers = 0;
    num_intertrainer_barriers = 0;
    num_global_barriers = 0;
    bytes_sent = 0;
    bytes_received = 0;
  }

  /** Return true if mat can be transmitted. */
  static inline bool is_sendable(const AbsMat& mat)
  {
    // This assumes we do not transmit mat with a datatype smaller than
    // DataType.
    // MPI uses "int" as its count type; do calculations with larger ints.
    size_t count = (size_t)mat.Height() * (size_t)mat.Width();
    return count <= (size_t)std::numeric_limits<int>::max();
  }
  /** Return true if the local portion of dist_mat can be transmitted. */
  static inline bool is_sendable(const AbsDistMat& dist_mat)
  {
    return is_sendable(dist_mat.LockedMatrix());
  }

  /** Return the intertrainer communicator. */
  const El::mpi::Comm& get_intertrainer_comm() const
  {
    return intertrainer_comm;
  }

  /** Return the trainer communicator. */
  const El::mpi::Comm& get_trainer_comm() const { return trainer_comm; }

  /** Return the world communicator. */
  const El::mpi::Comm& get_world_comm() const { return world_comm; }

  /** Return the communicator for this node. */
  const El::mpi::Comm& get_node_comm() const { return node_comm; }

  /**
   * Return a communicator containing num_per_group processors.
   *
   * This will attempt to pack processes so that the processes in each group
   * are physically close together on the system.
   *
   * num_per_group must evenly divide the number of processors in the world.
   */
  const El::mpi::Comm& get_packed_group_comm(int num_per_group) const;

  /** Return true if rank (in comm) is on the local node. */
  bool is_rank_node_local(int rank, const El::mpi::Comm& comm) const
  {
    // Translating to COMM_WORLD is typically constant time.
    int world_rank = El::mpi::Translate(comm, rank, get_world_comm());
    return is_world_rank_on_node(world_rank);
  }

  /** throws an lbann_exception **/
  void lbann_comm_abort(std::string msg);

private:
  /** World communicator. */
  const El::mpi::Comm world_comm;
  /** Communicator for every process in this trainer. */
  El::mpi::Comm trainer_comm;
  /** Communicator for every process with the same trainer rank. */
  El::mpi::Comm intertrainer_comm;
  /** Communicator for every process in the same compute node. */
  El::mpi::Comm node_comm;
  /** Packed group communicators. */
  mutable std::unordered_map<int, El::mpi::Comm> group_communicators;
  /** Grid for this trainer. */
  Grid* grid;
  /** Number of trainers. */
  int num_trainers;
  /** Number of processors per trainer. */
  int procs_per_trainer;
  /** Rank of the trainer this process is in. */
  int trainer_rank;
  /** Rank of this process within its trainer. */
  int rank_in_trainer;
  /** Number of processers per compute node. */
  int procs_per_node;
  /** Rank of this process within its compute node. */
  int rank_in_node;
  /** The list of world ranks that are on this compute node. */
  std::vector<int> world_ranks_on_node;
  /** Default number of threads per process.
   *  This is the number of OpenMP threads to use for parallel
   *  regions, provided omp_set_num_threads has not been called or the
   *  num_threads directive has not been provided.
   */
  int threads_per_proc;

  // Various statistics counters.
  size_t num_trainer_barriers;
  size_t num_intertrainer_barriers;
  size_t num_global_barriers;
  size_t bytes_sent;
  size_t bytes_received;

  /** Setup communicator for processes in the same compute node. */
  void setup_node_comm();

  /** Initialize the default number of threads per process.
   *  This is the number of OpenMP threads to use for parallel
   *  regions, provided omp_set_num_threads has not been called or the
   *  num_threads directive has not been provided. If the environment
   *  variable OMP_NUM_THREADS is defined, it's value is used for the
   *  default. Otherwise, then the default is the number of hardware
   *  cores per node divided by the number of processes per node.
   */
  void setup_threads();
};

/** Get the current rank within MPI_COMM_WORLD.
 *  This function is safe to call even if MPI has not initialized or
 *  has been finalized. In either case it returns a negative value.
 */
int get_rank_in_world();

} // namespace lbann

#endif // LBANN_COMM_HPP_INCLUDED
