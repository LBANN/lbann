////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <lbannv2_config.h>

/**
 * @file Enable spdlog logging for LBANNv2.
 *
 * The symbols in this file are not exported by default so any
 * hypothetical downstream doesn't take over our logger.
 *
 * The logger macros that include `LOG` in their names take a logger
 * pointer as their first argument. The other macros use the default
 * LBANNv2 logger.
 */

#include <spdlog/spdlog.h>

// These dispatch through SPDLOG's default macros. Hence, their
// behavior is ultimately determined by the SPDLOG_ACTIVE_LEVEL macro.
#define LBANNV2_LOG_TRACE(logger, ...) SPDLOG_LOGGER_TRACE(logger, __VA_ARGS__)
#define LBANNV2_LOG_DEBUG(logger, ...) SPDLOG_LOGGER_DEBUG(logger, __VA_ARGS__)
#define LBANNV2_LOG_INFO(logger, ...) SPDLOG_LOGGER_INFO(logger, __VA_ARGS__)
#define LBANNV2_LOG_WARN(logger, ...) SPDLOG_LOGGER_WARN(logger, __VA_ARGS__)
#define LBANNV2_LOG_ERROR(logger, ...) SPDLOG_LOGGER_ERROR(logger, __VA_ARGS__)
#define LBANNV2_LOG_CRITICAL(logger, ...)                                      \
  SPDLOG_LOGGER_CRITICAL(logger, __VA_ARGS__)

#define LBANNV2_TRACE(...)                                                     \
  LBANNV2_LOG_TRACE(::lbannv2::default_logger(), __VA_ARGS__)
#define LBANNV2_DEBUG(...)                                                     \
  LBANNV2_LOG_DEBUG(::lbannv2::default_logger(), __VA_ARGS__)
#define LBANNV2_INFO(...)                                                      \
  LBANNV2_LOG_INFO(::lbannv2::default_logger(), __VA_ARGS__)
#define LBANNV2_WARN(...)                                                      \
  LBANNV2_LOG_WARN(::lbannv2::default_logger(), __VA_ARGS__)
#define LBANNV2_ERROR(...)                                                     \
  LBANNV2_LOG_ERROR(::lbannv2::default_logger(), __VA_ARGS__)
#define LBANNV2_CRITICAL(...)                                                  \
  LBANNV2_LOG_CRITICAL(::lbannv2::default_logger(), __VA_ARGS__)

namespace lbannv2
{
/** @brief Get LBANNv2's default logger.
 *
 *  The default logger is configured through the environment variable
 *  `LBANNV2_LOG_FILE`. Acceptable values are 'stdout', 'stderr', and
 *  a valid filename pattern.
 *
 *  @todo Enable logging to a process-specific file.
 */
std::shared_ptr<::spdlog::logger>& default_logger();
}  // namespace lbannv2
