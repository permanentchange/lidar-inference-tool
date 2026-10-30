#include <iostream>
#include <type_traits>
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>
namespace inference_tools {
struct Json {
  using Object = std::map<std::string, Json>;
  using Array = std::vector<Json>;
  std::variant<std::nullptr_t, bool, int64_t, uint64_t, double, std::string, Object, Array> value;
  Json(): value(nullptr) {} Json(std::nullptr_t): value(nullptr) {}
  Json(bool v): value(v) {} Json(int v): value(int64_t(v)) {}
  Json(int64_t v): value(v) {} Json(uint64_t v): value(v) {}
  Json(double v): value(v) {} Json(float v): value(double(v)) {}
  Json(const char* v): value(std::string(v)) {} Json(std::string v): value(std::move(v)) {}
  Json(Object v): value(std::move(v)) {} Json(Array v): value(std::move(v)) {}
  Json& operator[](const std::string& key) {
    if (!std::holds_alternative<Object>(value)) value = Object{};
    return std::get<Object>(value)[key];
  }
};
inline std::string EscapeJson(const std::string& s) {
  std::ostringstream out; out << '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  out << '"'; return out.str();
}
inline std::ostream& operator<<(std::ostream& out, const Json& j) {
  std::visit([&](const auto& v) {
    using T = std::decay_t<decltype(v)>;
    if constexpr (std::is_same_v<T, std::nullptr_t>) out << "null";
    else if constexpr (std::is_same_v<T, bool>) out << (v ? "true" : "false");
    else if constexpr (std::is_same_v<T, std::string>) out << EscapeJson(v);
    else if constexpr (std::is_same_v<T, Json::Object>) {
      out << '{'; bool first = true; for (const auto& [k, x] : v) {
        if (!first) out << ','; first = false; out << EscapeJson(k) << ':' << x;
      } out << '}';
    } else if constexpr (std::is_same_v<T, Json::Array>) {
      out << '['; bool first = true; for (const auto& x : v) {
        if (!first) out << ','; first = false; out << x;
      } out << ']';
    } else if constexpr (std::is_same_v<T, double>) {
      if (!std::isfinite(v)) throw std::runtime_error("Nonfinite JSON number");
      out << std::setprecision(17) << v;
    } else out << v;
  }, j.value); return out;
}
inline void WriteJson(const std::string& path, const Json& j) {
  std::ofstream out(path); if (!out) throw std::runtime_error("Cannot write " + path);
  out << j << '\n'; if (!out) throw std::runtime_error("Write failed: " + path);
}
struct TimingStats {
  size_t n = 0; double mean = 0, min = 0, max = 0, p50 = 0, p90 = 0;
  Json json() const { return Json::Object{{"n", uint64_t(n)}, {"mean", mean}, {"min", min},
    {"max", max}, {"p50", p50}, {"p90", p90}}; }
};
inline TimingStats ComputeStats(std::vector<double> values) {
  TimingStats s; s.n = values.size(); if (values.empty()) return s;
  for (auto v : values) { if (!std::isfinite(v) || v < 0) throw std::runtime_error("Invalid timing sample"); s.mean += v; }
  s.mean /= s.n; std::sort(values.begin(), values.end()); s.min = values.front(); s.max = values.back();
  auto percentile = [&](double p) { double i = p * (values.size() - 1); size_t lo = size_t(i), hi = std::min(lo+1, values.size()-1);
    return values[lo] + (values[hi]-values[lo]) * (i-lo); };
  s.p50 = percentile(.5); s.p90 = percentile(.9); return s;
}
inline void PrintStats(const std::string& name, const std::string& clock, const TimingStats& s) {
  std::cout << std::left << std::setw(28) << name << std::setw(7) << clock << std::right
    << std::setw(6) << s.n << std::fixed << std::setprecision(3)
    << std::setw(10) << s.mean << std::setw(10) << s.min << std::setw(10) << s.max
    << std::setw(10) << s.p50 << std::setw(10) << s.p90 << '\n';
}
}  // namespace inference_tools
