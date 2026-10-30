#pragma once

#include <ostream>
#include <stdexcept>
#include <string>

namespace inference_tools {

enum class ToolKind { kInfer, kPerf };
enum class ParseResult { kOk, kHelp, kError };

struct CliOptions {
  std::string image_input = "raw";
  bool image_input_explicit = false;
  bool dump_result = true;
  std::string model_dir;
  std::string input_path;
  std::string dump_dir;
  std::string override_config_path;
  int source_num_point_features = 0;
  bool dump_decoded = true;
  bool dump_internal = true;
  int warmup_iters = 0;
  int profile_iters = 1;
};

inline CliOptions DefaultCliOptions(ToolKind kind) {
  CliOptions options;
  if (kind == ToolKind::kPerf) {
    options.dump_result = false;
    options.dump_decoded = false;
    options.dump_internal = false;
    options.warmup_iters = 10;
    options.profile_iters = 50;
  }
  return options;
}

inline void PrintUsage(std::ostream& output, const char* program, ToolKind kind) {
  const CliOptions defaults = DefaultCliOptions(kind);
  output << "Usage: " << program
         << " --model-dir MODEL_DIR --input INPUT_PATH --dump-dir DUMP_DIR"
         << " [--source-num-point-features N] [--override-config PATH]"
         << " [--warmup N] [--profile-iters N] [--image-input raw|preprocessed]";
  output << " [--dump-result|--no-dump-result] [--dump-decoded|--no-dump-decoded] [--dump-internal|--no-dump-internal]";
  output << "\n\nRequired paths:\n"
         << "  --model-dir MODEL_DIR          Model directory containing model.yaml\n"
         << "  --input INPUT_PATH            "
         << (kind == ToolKind::kInfer ? "PCD/BIN or BEVFusion manifest; directory supported\n"
                                      : "Single PCD/BIN file or BEVFusion manifest\n")
         << "  --dump-dir DUMP_DIR            Output directory (required; no default)\n"
         << "\nInput format:\n"
         << "  --source-num-point-features N  BIN source features: 3, 4, or 5\n"
         << "                                Default: auto-detect unambiguous BIN layout; PCD uses its header\n"
         << "  --image-input MODE            BEVFusion raw JPEG (default: raw) or preprocessed NPY\n"
         << "\nConfiguration:\n"
         << "  --override-config PATH        Package YAML override (default: none)\n"
         << "\nTiming:\n"
         << "  --warmup N                    Warmup iterations >= 0 (default: "
         << defaults.warmup_iters << ")\n"
         << "  --profile-iters N             Profile iterations > 0 (default: "
         << defaults.profile_iters << ")\n";
  output << "\nDebug dumps (outside timing; infer defaults on, perf defaults off):\n"
    << "  --dump-result / --no-dump-result     Final result JSON, arrays and input points\n"
    << "  --dump-decoded / --no-dump-decoded   Decoded arrays\n"
    << "  --dump-internal / --no-dump-internal Internal tensors\n"
    << "  Repeated switches: last occurrence wins\n"
    << "\nPerf collects normal total latency and stage analysis in separate passes;\n"
    << "each pass uses the requested warmup/profile iterations. Infer measures total only.\n";
  output << "\nHelp:\n"
         << "  -h, --help                    Show this help\n";
}

inline bool ParseInt(const std::string& value, int* out_value) {
  try {
    size_t parsed = 0;
    const int result = std::stoi(value, &parsed);
    if (parsed != value.size()) {
      return false;
    }
    *out_value = result;
    return true;
  } catch (const std::invalid_argument&) {
    return false;
  } catch (const std::out_of_range&) {
    return false;
  }
}

inline ParseResult ParseCliOptions(int argc, char** argv, ToolKind kind,
                                   CliOptions* out_options, std::string* out_error) {
  CliOptions options = DefaultCliOptions(kind);
  const auto fail = [out_error](const std::string& error) {
    if (out_error != nullptr) {
      *out_error = error;
    }
    return ParseResult::kError;
  };
  for (int index = 1; index < argc;) {
    const std::string arg = argv[index++];
    if (arg == "--help" || arg == "-h") {
      return ParseResult::kHelp;
    }
    if (arg == "--dump-result" || arg == "--no-dump-result") {
      options.dump_result = arg == "--dump-result"; continue;
    }
    if ((arg == "--dump-decoded" || arg == "--no-dump-decoded")) {
      options.dump_decoded = (arg == "--dump-decoded");
      continue;
    }
    if ((arg == "--dump-internal" || arg == "--no-dump-internal")) {
      options.dump_internal = (arg == "--dump-internal");
      continue;
    }
    const bool path_option = arg == "--model-dir" || arg == "--input" ||
                             arg == "--dump-dir" || arg == "--override-config" || arg == "--image-input";
    const bool count_option = arg == "--warmup" || arg == "--profile-iters" ||
                              arg == "--source-num-point-features";
    if (!path_option && !count_option) {
      if (arg == "--profile") {
        return fail("Unknown argument: --profile; use --profile-iters N instead");
      }
      if (!arg.empty() && arg[0] == '-') {
        return fail("Unknown argument: " + arg);
      }
      return fail("Unexpected positional argument: " + arg +
                  "; use --model-dir MODEL_DIR --input INPUT_PATH --dump-dir DUMP_DIR");
    }
    if (index >= argc || std::string(argv[index]).rfind("--", 0) == 0 ||
        std::string(argv[index]) == "-h") {
      return fail(std::string("Missing ") + (path_option ? "path" : "count") + " for " + arg);
    }
    const std::string value = argv[index++];
    if (path_option) {
      if (value.empty()) {
        return fail("Missing path for " + arg);
      }
      if (arg == "--model-dir") {
        options.model_dir = value;
      } else if (arg == "--input") {
        options.input_path = value;
      } else if (arg == "--dump-dir") {
        options.dump_dir = value;
      } else if (arg == "--image-input") {
        if (value != "raw" && value != "preprocessed") return fail("--image-input must be raw or preprocessed");
        options.image_input = value;
        options.image_input_explicit = true;
      } else {
        options.override_config_path = value;
      }
    } else {
      int count = 0;
      if (!ParseInt(value, &count)) {
        return fail("Invalid integer for " + arg + ": " + value);
      }
      if (arg == "--warmup") {
        if (count < 0) {
          return fail("--warmup must be >= 0");
        }
        options.warmup_iters = count;
      } else if (arg == "--profile-iters") {
        if (count <= 0) {
          return fail("--profile-iters must be > 0");
        }
        options.profile_iters = count;
      } else {
        if (count < 3 || count > 5) {
          return fail("--source-num-point-features must be one of 3, 4, or 5");
        }
        options.source_num_point_features = count;
      }
    }
  }
  if (options.model_dir.empty()) {
    return fail("--model-dir is required");
  }
  if (options.input_path.empty()) {
    return fail("--input is required");
  }
  if (options.dump_dir.empty()) {
    return fail("--dump-dir is required");
  }
  *out_options = options;
  return ParseResult::kOk;
}

}  // namespace inference_tools
