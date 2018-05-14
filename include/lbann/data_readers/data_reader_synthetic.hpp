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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_SYNTHETIC_HPP
#define LBANN_DATA_READER_SYNTHETIC_HPP

#include "data_reader.hpp"

namespace lbann {
class data_reader_synthetic : public generic_data_reader {
 public:
  //@todo: add what data distribution to use
  data_reader_synthetic(int num_samples, int num_features, bool shuffle = true);
  data_reader_synthetic(const data_reader_synthetic&) = default;
  data_reader_synthetic& operator=(const data_reader_synthetic&) = default;
  ~data_reader_synthetic() override {}
  data_reader_synthetic* copy() const override {
    return new data_reader_synthetic(*this);
  }
  std::string get_type() const override {
    return "data_reader_synthetic";
  }

  void load() override;

  int get_num_samples() const {
    return m_num_samples;
  }
  int get_num_features() const {
    return m_num_features;
  }

  int get_linearized_data_size() const override {
    return m_num_features;
  }

  const std::vector<int> get_data_dims() const override {
    return {m_num_features};
  }

 protected:
  bool fetch_datum(CPUMat& X, int data_id, int mb_idx, int tid) override;

 private:
  int  m_num_samples; //rows
  int  m_num_features; //cols
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_SYNTHETIC_HPP
