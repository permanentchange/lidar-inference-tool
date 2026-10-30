#include "debug_runner.hpp"
int main(int argc, char** argv) {
  return inference_tools::RunDebugTool(argc, argv, inference_tools::ToolKind::kInfer);
}
