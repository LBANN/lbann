////////////////////////////////////////////////////////////////////////////////
// Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
// LBANN Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include "lbannv2/utils/logging.hpp"

#include <memory>
#include <string>

#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#if __has_include(<unistd.h>)
#include <unistd.h>
#define _HAVE_UNISTD_H
#endif

namespace
{
spdlog::level::level_enum get_env_log_level()
{
  if (char const* const var = std::getenv("LBANNV2_LOG_LEVEL"))
  {
    std::string level_str {var};
    std::for_each(begin(level_str), end(level_str), [](char& c) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    });

    if (level_str == "trace")
      return ::spdlog::level::trace;
    if (level_str == "debug")
      return ::spdlog::level::debug;
    if (level_str == "info")
      return ::spdlog::level::info;
    if (level_str == "warn")
      return ::spdlog::level::warn;
    if (level_str == "err")
      return ::spdlog::level::err;
    if (level_str == "critical")
      return ::spdlog::level::critical;
    if (level_str == "off")
      return ::spdlog::level::off;
  }
  return ::spdlog::level::info;
}

std::string get_hostname()
{
#ifdef _HAVE_UNISTD_H
  char buf[256];
  if (gethostname(buf, 256) == 0)
    return std::string {buf, std::find(buf, buf + 256, '\0')};
#endif

  return "<unknownhost>";
}

// The one in H2 is not exported, but it's a quick reimplementation.
class HostFlag final : public spdlog::custom_flag_formatter
{
public:
  std::unique_ptr<custom_flag_formatter> clone() const final
  {
    return spdlog::details::make_unique<HostFlag>();
  }
  void format(::spdlog::details::log_msg const&,
              ::std::tm const&,
              ::spdlog::memory_buf_t& dest)
  {
    static std::string const hostname = get_hostname();
    dest.append(hostname);
  }
};  // class HostFlag

std::unique_ptr<::spdlog::pattern_formatter> make_default_formatter()
{
  auto formatter = std::make_unique<::spdlog::pattern_formatter>();
  formatter->add_flag<HostFlag>('h');
  formatter->set_pattern("[%h:%P:%t] [%n:%^%l%$] %v");
  // formatter->set_pattern("[%m-%d-%Y %T.%f] [%h:%P] [%n] [%^%l%$] %v");
  return formatter;
}

::spdlog::sink_ptr make_default_sink()
{
  char const* sink_name = std::getenv("LBANNV2_LOG_FILE");
  std::string const sink_name_str(sink_name ? sink_name : "stdout");
  if (sink_name_str == "stdout")
    return std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  if (sink_name_str == "stderr")
    return std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
  return std::make_shared<spdlog::sinks::basic_file_sink_mt>(sink_name_str);
}

std::shared_ptr<::spdlog::logger> make_default_logger()
{
  auto logger =
    std::make_shared<::spdlog::logger>("lbannv2", make_default_sink());
  logger->set_formatter(make_default_formatter());
  logger->set_level(get_env_log_level());
  return logger;
}

}  // namespace

std::shared_ptr<::spdlog::logger>& lbannv2::default_logger()
{
  static std::shared_ptr<::spdlog::logger> logger_ = make_default_logger();
  return logger_;
}
