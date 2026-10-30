#include <bevfusion_inference.hpp>
#include <iostream>
#include <stdexcept>
int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: bevfusion_cpp_api_example MODEL_DIR SAMPLE_MANIFEST\n";
    return 1;
  }
  try {
    inference::BevFusionInferenceRuntime runtime(argv[1]);
    auto frame = inference::LoadBevFusionFrame(argv[2], runtime.config());
    auto result = runtime.Infer(frame);
    for (const auto& d : result.detections)
      std::cout << d.label_name << ' ' << d.score << " center=" << d.box[0]
                << ',' << d.box[1] << ',' << d.box[2]
                << " velocity=" << d.velocity[0] << ',' << d.velocity[1]
                << '\n';
    std::cout << "detections: " << result.detections.size() << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
