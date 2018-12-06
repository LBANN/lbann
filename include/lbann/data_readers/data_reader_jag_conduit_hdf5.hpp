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

#ifndef _DATA_READER_JAG_CONDUIT_HDF5_HPP_
#define _DATA_READER_JAG_CONDUIT_HDF5_HPP_

#include "lbann_config.hpp" // may define LBANN_HAS_CONDUIT

#ifdef LBANN_HAS_CONDUIT
#include "lbann/data_readers/opencv.hpp"
#include "data_reader.hpp"
#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "lbann/data_readers/cv_process.hpp"
#include <string>
#include <set>
#include <unordered_map>

namespace lbann {

class jag_store;

/**
 * Loads the pairs of JAG simulation inputs and results from a conduit-wrapped hdf5 file
 */
class data_reader_jag_conduit_hdf5 : public generic_data_reader {
 public:
  using ch_t = float; ///< jag output image channel type
  using scalar_t = double; ///< jag scalar output type
  using input_t = double; ///< jag input parameter type

  /**
   * Dependent/indepdendent variable types
   * - JAG_Image: simulation output images
   * - JAG_Scalar: simulation output scalars
   * - JAG_Input: simulation input parameters
   * - Undefined: the default
   */
  enum variable_t {Undefined=0, JAG_Image, JAG_Scalar, JAG_Input};
  using TypeID = conduit::DataType::TypeID;

  data_reader_jag_conduit_hdf5(bool shuffle = true) = delete;
  data_reader_jag_conduit_hdf5(const std::shared_ptr<cv_process>& pp, bool shuffle = true);
  data_reader_jag_conduit_hdf5(const data_reader_jag_conduit_hdf5&);
  data_reader_jag_conduit_hdf5& operator=(const data_reader_jag_conduit_hdf5&);
  ~data_reader_jag_conduit_hdf5() override;
  data_reader_jag_conduit_hdf5* copy() const override { return new data_reader_jag_conduit_hdf5(*this); }

  std::string get_type() const override {
    return "data_reader_jag_conduit_hdf5";
  }

  /// Load data and do data reader's chores.
  void load() override;

  /// Return the number of samples
  size_t get_num_samples() const;

  /// Return the number of measurement views
  unsigned int get_num_img_srcs() const;
  // Return the number of channels in an image
  unsigned int get_num_channels() const;
  /// Return the linearized size of an image;
  size_t get_linearized_image_size() const;
  /// Return the linearized size of one channel in the image
  size_t get_linearized_channel_size() const;
  /// Return the linearized size of scalar outputs
  size_t get_linearized_scalar_size() const;
  /// Return the linearized size of inputs
  size_t get_linearized_input_size() const;

  /// Return the total linearized size of data
  int get_linearized_data_size() const override;
  /// Return the total linearized size of response
  int get_linearized_response_size() const override;
  /// Return the per-source linearized sizes of composite data
  std::vector<size_t> get_linearized_data_sizes() const;
  /// Return the per-source linearized sizes of composite response
  std::vector<size_t> get_linearized_response_sizes() const;

  /// Return the dimension of data
  const std::vector<int> get_data_dims() const override;

  int get_num_labels() const override;
  int get_linearized_label_size() const override;

  /// Show the description
  std::string get_description() const;

  /// Return the image simulation output of the i-th sample
  std::vector<cv::Mat> get_cv_images(const size_t i, int tid) const;

  template<typename S>
  static size_t add_val(const std::string key, const conduit::Node& n, std::vector<S>& vals);

  /// sets up a data_store.
  void setup_data_store(model *m) override;

  /// A untiliy function to convert the pointer to image data into an opencv image
  static cv::Mat cast_to_cvMat(const std::pair<size_t, const ch_t*> img, const int height);

  void set_image_dims(const int width, const int height, const int ch=1);

  void set_scalar_keys(const std::string &keys) { m_scalar_keys = keys; }
  void set_input_keys(const std::string &keys) { m_input_keys = keys; }
  void set_image_views(const std::string &views) { m_image_views = views; }
  void set_image_channels(const std::string &channels) { m_image_channels = channels; }

  void post_update() override;

 protected:

  friend jag_store;

  virtual void set_defaults();
  virtual bool replicate_processor(const cv_process& pp);
  virtual void copy_members(const data_reader_jag_conduit_hdf5& rhs);

  bool fetch_datum(CPUMat& X, int data_id, int mb_idx, int tid); 

  virtual std::vector<CPUMat>
    create_datum_views(CPUMat& X, const std::vector<size_t>& sizes, const int mb_idx) const;

  bool fetch_label(CPUMat& X, int data_id, int mb_idx, int tid) override;

  /// Check if the given sample id is valid
  bool check_sample_id(const size_t i) const;

  /// Choose the image closest to the bang time among those associated with the i-th sample
  std::vector<int> choose_image_near_bang_time(const size_t i) const;

  jag_store * get_jag_store() const { return m_jag_store; }

  int m_image_width; ///< image width
  int m_image_height; ///< image height
  int m_image_num_channels; ///< number of image channels

  /// Whether data have been loaded
  bool m_is_data_loaded;

  int m_num_labels; ///< number of labels

  /// preprocessor duplicated for each omp thread
  std::vector<std::unique_ptr<cv_process> > m_pps;

  /// jag_store; replaces m_data
  jag_store *m_jag_store;

  bool m_owns_jag_store;

  /**
   * Set of keys that are associated with non_numerical values.
   * Such a variable requires a specific method for mapping to a numeric value.
   * When a key is found in the set, the variable is ignored. Therefore,
   * when a conversion is defined for such a key, remove it from the set.
   */
  static const std::set<std::string> non_numeric_vars;

  /**
   * indicate if all the input variables are of the input_t type, in which case
   * we can rely on a data extraction method with lower overhead.
   */
  bool m_uniform_input_type;

  /**
   * maps integers to sample IDs. In the future the sample IDs may
   * not be integers; also, this map only includes sample IDs that
   * have <sample_id>/performance/success = 1
   */
  std::unordered_map<int, std::string> m_success_map;

  std::set<std::string> m_emi_selectors;

  std::string m_scalar_keys;
  std::string m_input_keys;
  std::string m_image_views;
  std::string m_image_channels;

  data_reader_jag_conduit_hdf5* m_primary_reader;
};



} // end of namespace lbann
#endif // LBANN_HAS_CONDUIT
#endif // _DATA_READER_JAG_CONDUIT_HDF5_HPP_
