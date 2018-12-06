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

#ifndef __LBANN_CALLBACKS_CALLBACK_LTFB_HPP_INCLUDED
#define __LBANN_CALLBACKS_CALLBACK_LTFB_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

namespace lbann {

/**
 * Manage LTFB training.
 * LTFB works in rounds, which are made up of some number of mini-batches (that
 * evenly divide the number of minibatches in an epoch). In each round, the
 * model trains as usual, and at the end it is randomly paired with another
 * model. The pairs exchange their models and evaluate both their local and the
 * received model on their validation data. The model achieving the highest
 * accuracy is retained and training continues.
 * Extension to GAN list of weights to send are specified
 * For example, a trainer will evaluate on its generator and partner's generator
 * using its holdout tournament data and local discriminator
 * Current limitations:
 * - Does not transfer optimizer state, so it's best to stick to SGD without
 * momentum.
 * - Uses the validation data for the tournament (we may not want this).
 * - Requires a manually-created model duplicate.
 */
class lbann_callback_ltfb : public lbann_callback {
 public:

  /** Constructor.
   *  @param round_size The number of minibatches in each round.
   *  @param increasing_metric_mode  The expectation for a good tournament metric, 
   *  default, increasing trend is good  
   *  @todo pair metric_mode with eval_metric
   *  @param eval_metric Tournament evaluation metrics
   *  @param selected_weights set of weights to exchange
   */
  lbann_callback_ltfb(int round_size, 
                      std::unordered_set<std::string> eval_metrics,
                      bool increasing_metric_mode = true,
                      std::unordered_set<std::string> weights_tosend = std::unordered_set<std::string>(),
                      lbann_summary* summarizer = nullptr);
  lbann_callback_ltfb(const lbann_callback_ltfb& other);
  lbann_callback_ltfb& operator=(const lbann_callback_ltfb& other);
  ~lbann_callback_ltfb() override;
  lbann_callback_ltfb* copy() const override { return new lbann_callback_ltfb(*this); }
  std::string name() const override { return "ltfb"; }

  /** Set up LTFB. */
  void setup(model *m) override;
  /** Potentially run an LTFB round. */
  void on_batch_begin(model *m) override;

 private:

  /** LBANN communicator. */
  lbann_comm *m_comm;
  /** Number of minibatches in a round. */
  int m_round_size;
  /** Evaluation metrics. */
  std::unordered_set<std::string> m_eval_metrics;
  /** Flag to determine expectation for a good tournament metric: default is increasing */
  bool m_increasing_metric_mode;
  /** List of weights to send. */
  std::unordered_set<std::string> m_weights_tosend;
  /** Weights from local model. */
  std::vector<weights*> m_local_weights;

};

}  // namespace lbann

#endif  // __LBANN_CALLBACKS_CALLBACK_LTFB_HPP_INCLUDED
