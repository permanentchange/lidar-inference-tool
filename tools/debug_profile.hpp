#pragma once
#include <stdexcept>
namespace inference_tools {
// Warmups are discarded; there are no implicit initialization inferences.
template <class Infer, class Consume>
void RunProfilePass(int warmup, int iterations, Infer infer, Consume consume) {
  if (warmup < 0 || iterations <= 0) throw std::invalid_argument("Invalid profile counts");
  for (int i=0;i<warmup;++i) infer();
  for (int i=0;i<iterations;++i) consume(infer());
}
}
