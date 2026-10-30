#pragma once
#include <algorithm>
#include <filesystem>
#include <set>
#include <vector>

#include "debug_cli.hpp"
namespace inference_tools {
inline void SortBevFusionManifests(std::vector<std::filesystem::path>* frames) {
  std::set<std::string> ids;
  bool numeric = true;
  for (const auto& path : *frames) {
    auto id = path.stem().string();
    if (!ids.insert(id).second)
      throw std::runtime_error("Duplicate manifest frame ID: " + id);
    numeric = numeric && !id.empty() &&
              id.find_first_not_of("0123456789") == std::string::npos;
  }
  std::sort(
      frames->begin(), frames->end(), [numeric](const auto& a, const auto& b) {
        if (numeric) {
          auto x = a.stem().string(), y = b.stem().string();
          auto xi = x.find_first_not_of('0'), yi = y.find_first_not_of('0');
          x = xi == std::string::npos ? "0" : x.substr(xi);
          y = yi == std::string::npos ? "0" : y.substr(yi);
          if (x.size() != y.size()) return x.size() < y.size();
          if (x != y) return x < y;
        }
        return a.filename() < b.filename();
      });
}
}  // namespace inference_tools
