#include "inference_cpp.hpp"

#include <type_traits>
#include <utility>

static_assert(std::is_same<
              decltype(std::declval<inference::PostprocessConfig>().self_car_box),
              std::optional<std::array<float, inference::kBoxDimension>>>::value,
              "CUDA compilation must support the C++17 public API");
