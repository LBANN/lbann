////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
//
// collective_test.cpp - Tests custom LBANN collective implementations
////////////////////////////////////////////////////////////////////////////////

#include <cstdlib>
#include "lbann/comm.hpp"
#include "lbann/utils/timer.hpp"
#include "test_utils.hpp"

using namespace lbann;

const IntType num_trials = 20;

void add_buffer_into_mat(const uint8_t *buf_, Mat& accum) {
  const IntType height = accum.Height();
  const IntType width = accum.Width();
  const auto *buf = (const DataType *) buf_;
  DataType *accum_buf = accum.Buffer();
  for (IntType i = 0; i < height*width; ++i) {
    accum_buf[i] += buf[i];
  }
}

void test_rd_allreduce(lbann_comm *comm, DistMat& dmat) {
  auto send_transform =
    [] (Mat& mat, El::IR h, El::IR w, IntType& send_size, bool const_data,
  IntType call_idx) {
    auto to_send = mat(h, w);
    send_size = sizeof(DataType) * to_send.Height() * to_send.Width();
    return (uint8_t *) to_send.Buffer();
  };
  auto recv_apply_transform =
  [] (uint8_t *recv_buf, Mat& accum, bool is_local) {
    add_buffer_into_mat(recv_buf, accum);
    return sizeof(DataType) * accum.Height() * accum.Width();
  };
  Mat& mat = dmat.Matrix();
  IntType max_recv_count = sizeof(DataType) * mat.Height() * mat.Width();
  comm->recursive_doubling_allreduce_pow2(
    comm->get_intermodel_comm(), mat, max_recv_count,
    std::function<uint8_t *(Mat&, El::IR, El::IR, IntType&, bool, IntType)>(send_transform),
    std::function<IntType(uint8_t *, Mat&, bool)>(recv_apply_transform), {});
}

void test_pe_ring_allreduce(lbann_comm *comm, DistMat& dmat) {
  auto send_transform =
    [] (Mat& mat, El::IR h, El::IR w, IntType& send_size, bool const_data,
  IntType call_idx) {
    auto to_send = mat(h, w);
    send_size = sizeof(DataType) * to_send.Height() * to_send.Width();
    return (uint8_t *) to_send.Buffer();
  };
  auto recv_transform =
  [] (uint8_t *recv_buf, Mat& accum) {
    Mat recv_mat;
    recv_mat.LockedAttach(accum.Height(), accum.Width(), (DataType *) recv_buf,
                          accum.LDim());
    accum = recv_mat;
    return sizeof(DataType) * recv_mat.Height() * recv_mat.Width();
  };
  auto recv_apply_transform =
  [] (uint8_t *recv_buf, Mat& accum, bool is_local) {
    add_buffer_into_mat(recv_buf, accum);
    return sizeof(DataType) * accum.Height() * accum.Width();
  };
  Mat& mat = dmat.Matrix();
  IntType max_recv_count = sizeof(DataType) * mat.Height() * mat.Width();
  lbann_comm::allreduce_options opts;
  opts.id_recv = true;
  comm->pe_ring_allreduce(
    comm->get_intermodel_comm(), mat, max_recv_count,
    std::function<uint8_t *(Mat&, El::IR, El::IR, IntType&, bool, IntType)>(send_transform),
    std::function<IntType(uint8_t *, Mat&)>(recv_transform),
    std::function<IntType(uint8_t *, Mat&, bool)>(recv_apply_transform), opts);
}

void test_ring_allreduce(lbann_comm *comm, DistMat& dmat) {
  auto send_transform =
    [] (Mat& mat, El::IR h, El::IR w, IntType& send_size, bool const_data,
  IntType call_idx) {
    auto to_send = mat(h, w);
    send_size = sizeof(DataType) * to_send.Height() * to_send.Width();
    return (uint8_t *) to_send.Buffer();
  };
  auto recv_transform =
  [] (uint8_t *recv_buf, Mat& accum) {
    Mat recv_mat;
    recv_mat.LockedAttach(accum.Height(), accum.Width(), (DataType *) recv_buf,
                          accum.LDim());
    accum = recv_mat;
    return sizeof(DataType) * recv_mat.Height() * recv_mat.Width();
  };
  auto recv_apply_transform =
  [] (uint8_t *recv_buf, Mat& accum, bool) {
    add_buffer_into_mat(recv_buf, accum);
    return sizeof(DataType) * accum.Height() * accum.Width();
  };
  Mat& mat = dmat.Matrix();
  IntType max_recv_count = sizeof(DataType) * mat.Height() * mat.Width();
  lbann_comm::allreduce_options opts;
  opts.id_recv = true;
  comm->ring_allreduce(
    comm->get_intermodel_comm(), mat, max_recv_count,
    std::function<uint8_t *(Mat&, El::IR, El::IR, IntType&, bool, IntType)>(send_transform),
    std::function<IntType(uint8_t *, Mat&)>(recv_transform),
    std::function<IntType(uint8_t *, Mat&, bool)>(recv_apply_transform), opts);
}

void test_rabenseifner_allreduce(lbann_comm *comm, DistMat& dmat) {
  auto send_transform =
    [] (Mat& mat, El::IR h, El::IR w, IntType& send_size, bool const_data,
  IntType call_idx) {
    auto to_send = mat(h, w);
    send_size = sizeof(DataType) * to_send.Height() * to_send.Width();
    return (uint8_t *) to_send.Buffer();
  };
  auto recv_transform =
  [] (uint8_t *recv_buf, Mat& accum) {
    Mat recv_mat;
    recv_mat.LockedAttach(accum.Height(), accum.Width(), (DataType *) recv_buf,
                          accum.LDim());
    accum = recv_mat;
    return sizeof(DataType) * recv_mat.Height() * recv_mat.Width();
  };
  auto recv_apply_transform =
  [] (uint8_t *recv_buf, Mat& accum, bool is_local) {
    add_buffer_into_mat(recv_buf, accum);
    return sizeof(DataType) * accum.Height() * accum.Width();
  };
  Mat& mat = dmat.Matrix();
  IntType max_recv_count = sizeof(DataType) * mat.Height() * mat.Width();
  lbann_comm::allreduce_options opts;
  opts.id_recv = true;
  comm->rabenseifner_allreduce(
    comm->get_intermodel_comm(), mat, max_recv_count,
    std::function<uint8_t *(Mat&, El::IR, El::IR, IntType&, bool, IntType)>(send_transform),
    std::function<IntType(uint8_t *, Mat&)>(recv_transform),
    std::function<IntType(uint8_t *, Mat&, bool)>(recv_apply_transform), opts);
}

void print_stats(const std::vector<double>& times) {
  double sum = std::accumulate(times.begin() + 1, times.end(), 0.0);
  double mean = sum / (times.size() - 1);
  auto minmax = std::minmax_element(times.begin() + 1, times.end());
  double sqsum = 0.0;
  for (auto t = times.begin() + 1; t != times.end(); ++t) {
    sqsum += (*t - mean) * (*t - mean);
  }
  double stdev = std::sqrt(sqsum / (times.size() - 1));
  std::cout << "\tMean: " << mean << std::endl;
  std::cout << "\tMin: " << *(minmax.first) << std::endl;
  std::cout << "\tMax: " << *(minmax.second) << std::endl;
  std::cout << "\tStdev: " << stdev << std::endl;
  std::cout << "\tRaw: ";
  for (const auto& t : times) {
    std::cout << t << ", ";
  }
  std::cout << std::endl;
}

int main(int argc, char **argv) {
  El::Initialize(argc, argv);
  auto *comm = new lbann_comm(1);
  for (IntType mat_size = 1; mat_size <= 16384; mat_size *= 2) {
    std::vector<double> mpi_times, rd_times, pe_ring_times, ring_times,
        rab_times;
    // First trial is a warmup.
    for (IntType trial = 0; trial < num_trials + 1; ++trial) {
      DistMat rd_mat(comm->get_model_grid());
      El::Uniform(rd_mat, mat_size, mat_size, DataType(0.0), DataType(1.0));
      DistMat exact_mat(rd_mat);
      DistMat pe_ring_mat(rd_mat);
      DistMat ring_mat(rd_mat);
      DistMat rab_mat(rd_mat);
      comm->global_barrier();
      // Baseline.
      double start = get_time();
      comm->intermodel_sum_matrix(exact_mat);
      mpi_times.push_back(get_time() - start);
      comm->global_barrier();
      // Recursive doubling.
      start = get_time();
      test_rd_allreduce(comm, rd_mat);
      rd_times.push_back(get_time() - start);
      ASSERT_MAT_EQ(rd_mat.Matrix(), exact_mat.Matrix());
      comm->global_barrier();
      // Pairwise-exchange/ring.
      start = get_time();
      test_pe_ring_allreduce(comm, pe_ring_mat);
      pe_ring_times.push_back(get_time() - start);
      ASSERT_MAT_EQ(pe_ring_mat.Matrix(), exact_mat.Matrix());
      // Ring.
      start = get_time();
      test_ring_allreduce(comm, ring_mat);
      ring_times.push_back(get_time() - start);
      ASSERT_MAT_EQ(ring_mat.Matrix(), exact_mat.Matrix());
      // Rabenseifner.
      start = get_time();
      test_rabenseifner_allreduce(comm, rab_mat);
      rab_times.push_back(get_time() - start);
      ASSERT_MAT_EQ(rab_mat.Matrix(), exact_mat.Matrix());
    }
    if (comm->am_world_master()) {
      std::cout << "MPI (" << mat_size << "x" << mat_size << "):" << std::endl;
      print_stats(mpi_times);
      std::cout << "RD (" << mat_size << "x" << mat_size << "):" << std::endl;
      print_stats(rd_times);
      std::cout << "PE/ring (" << mat_size << "x" << mat_size << "):" <<
                std::endl;
      print_stats(pe_ring_times);
      std::cout << "Ring (" << mat_size << "x" << mat_size << "):" << std::endl;
      print_stats(ring_times);
      std::cout << "Rabenseifner (" << mat_size << "x" << mat_size << "):" <<
                std::endl;
      print_stats(rab_times);
    }
  }
  delete comm;
  El::Finalize();
  return 0;
}
