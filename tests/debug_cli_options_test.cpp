#include "debug_cli.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using inference_tools::CliOptions;
using inference_tools::ParseResult;
using inference_tools::ToolKind;

void Check(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

CliOptions Parse(ToolKind kind, std::vector<std::string> arguments) {
  std::vector<char*> argv;
  for (std::string& argument : arguments) {
    argv.push_back(argument.data());
  }
  CliOptions options;
  std::string error;
  const auto result = inference_tools::ParseCliOptions(
      static_cast<int>(argv.size()), argv.data(), kind, &options, &error);
  if (result != ParseResult::kOk) {
    throw std::runtime_error("Valid CLI rejected: " + error);
  }
  return options;
}

std::vector<std::string> RequiredArguments() {
  return {"debug_tool", "--model-dir", "model dir", "--input", "cloud.pcd",
          "--dump-dir", "output dir"};
}

void TestDefaults() {
  const auto infer = Parse(ToolKind::kInfer, RequiredArguments());
  Check(infer.dump_decoded && infer.dump_internal, "Infer dumps must default to enabled");
  Check(infer.warmup_iters == 0 && infer.profile_iters == 1, "Infer timing defaults changed");
  Check(infer.source_num_point_features == 0, "Source layout must default to auto-detection");
  Check(infer.override_config_path.empty(), "Override must default to absent");

  const auto perf = Parse(ToolKind::kPerf, RequiredArguments());
  Check(!perf.dump_decoded && !perf.dump_internal, "Perf must not enable result dumps");
  Check(perf.warmup_iters == 10 && perf.profile_iters == 50, "Perf timing defaults changed");
}

void TestCommonOptionsInAnyOrder() {
  for (const auto kind : {ToolKind::kInfer, ToolKind::kPerf}) {
    const auto options = Parse(kind, {
        "debug_tool", "--profile-iters", "3", "--dump-dir", "output dir",
        "--source-num-point-features", "5", "--input", "cloud.bin", "--warmup", "0",
        "--override-config", "override config.yaml", "--model-dir", "model dir"});
    Check(options.model_dir == "model dir" && options.input_path == "cloud.bin" &&
              options.dump_dir == "output dir", "Required paths were parsed incorrectly");
    Check(options.override_config_path == "override config.yaml", "Override path was lost");
    Check(options.source_num_point_features == 5, "Source feature count was lost");
    Check(options.warmup_iters == 0 && options.profile_iters == 3, "Timing overrides were lost");
  }
}

void TestDumpSwitches() {
  auto arguments = RequiredArguments();
  arguments.insert(arguments.end(), {"--no-dump-decoded", "--no-dump-internal"});
  auto options = Parse(ToolKind::kInfer, arguments);
  Check(!options.dump_decoded && !options.dump_internal, "Dump opt-outs were ignored");

  arguments.push_back("--dump-decoded");
  options = Parse(ToolKind::kInfer, arguments);
  Check(options.dump_decoded && !options.dump_internal, "Dump switches must be independent");

  arguments.push_back("--dump-internal");
  options = Parse(ToolKind::kInfer, arguments);
  Check(options.dump_decoded && options.dump_internal, "Positive dump switches must re-enable dumps");

  arguments.insert(arguments.end(), {"--no-dump-decoded", "--no-dump-internal"});
  options = Parse(ToolKind::kInfer, arguments);
  Check(!options.dump_decoded && !options.dump_internal, "Last dump switch must win");
}

}  // namespace

int main() {
  try {
    TestDefaults();
    TestCommonOptionsInAnyOrder();
    TestDumpSwitches();
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
  return 0;
}
