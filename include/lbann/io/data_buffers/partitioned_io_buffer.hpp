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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_PARTITIONED_IO_BUFFER_HPP_INCLUDED
#define LBANN_PARTITIONED_IO_BUFFER_HPP_INCLUDED

#include "lbann/io/data_buffers/generic_io_buffer.hpp"

namespace lbann {

/**
 * Parallel I/O routines for managing partitioned minibatches
 */
class partitioned_io_buffer : public generic_io_buffer {
 public:
  partitioned_io_buffer(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers, int num_child_layers);
  partitioned_io_buffer(const partitioned_io_buffer& other);
  partitioned_io_buffer& operator=(const partitioned_io_buffer& other);
  ~partitioned_io_buffer() = default;
  partitioned_io_buffer* copy() const override { return new partitioned_io_buffer(*this); }

  std::string get_type() const override { return "partitioned"; }

  void fp_setup_data(El::Int cur_mini_batch_size, int idx) override;
  void setup_data(El::Int num_neurons, El::Int num_targets, El::Int max_mini_batch_size) override;

  int fetch_to_local_matrix(generic_data_reader *data_reader, execution_mode mode) override;
  void distribute_from_local_matrix(generic_data_reader *data_reader, execution_mode mode, AbsDistMat& sample, AbsDistMat& response) override;
  void distribute_from_local_matrix(generic_data_reader *data_reader, execution_mode mode, AbsDistMat& sample) override;
  bool update_data_set(generic_data_reader *data_reader, execution_mode mode) override;
  int num_samples_ready(execution_mode mode) override;

  void calculate_num_iterations_per_epoch_spanning_models(int max_mini_batch_size, generic_data_reader *data_reader) override;
  void calculate_num_iterations_per_epoch_single_model(int max_mini_batch_size, generic_data_reader *data_reader) override;
  int compute_max_num_parallel_readers(long data_set_size, int mini_batch_size, int requested_num_parallel_readers) const override;
  static int compute_max_num_parallel_readers(long data_set_size, int mini_batch_size, int requested_num_parallel_readers, const lbann_comm* comm);

  /** Input buffers
   *  Each matrix column corresponds to a flattened mini-batch sample
   *  or label or responase.
   */
  std::vector<std::unique_ptr<AbsDistMat>> m_input_buffers;
  int m_num_samples_fetched;
};
}

#endif  // LBANN_PARTITIONED_IO_BUFFER_HPP_INCLUDED
