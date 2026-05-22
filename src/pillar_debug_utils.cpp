#include "pillar_runtime.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace inference_internal {
namespace {

constexpr int kBoxDim = 7;
InferenceStatus SetError(std::string* out_error,
                         InferenceStatus status,
                         const std::string& message) {
  if (out_error != nullptr) {
    *out_error = message;
  }
  return status;
}

template <typename T>
std::string DtypeTag();

template <>
std::string DtypeTag<float>() {
  return "<f4";
}

template <>
std::string DtypeTag<int32_t>() {
  return "<i4";
}

std::string ShapeToNpyString(const std::vector<size_t>& shape) {
  std::ostringstream oss;
  oss << "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    oss << shape[i];
    if (shape.size() == 1) {
      oss << ",";
    } else if (i + 1 < shape.size()) {
      oss << ", ";
    }
  }
  oss << ")";
  return oss.str();
}

std::string ShapeToJsonArray(const std::vector<size_t>& shape) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream oss;
  for (const char ch : value) {
    switch (ch) {
      case '\\':
        oss << "\\\\";
        break;
      case '"':
        oss << "\\\"";
        break;
      case '\n':
        oss << "\\n";
        break;
      case '\r':
        oss << "\\r";
        break;
      case '\t':
        oss << "\\t";
        break;
      default:
        oss << ch;
        break;
    }
  }
  return oss.str();
}

std::string NormalizeNmsType(const std::string& value) {
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return normalized;
}

RuntimeConfig RuntimeConfigFromPackage(const inference::PackageConfig& package) {
  RuntimeConfig runtime;
  runtime.end2end_file = package.model.engine_path;
  runtime.point_cloud_range = package.model.point_cloud_range;
  runtime.post_center_range = package.model.post_center_range;
  runtime.pillar_size = package.model.pillar_size;
  runtime.max_points_per_pillar = package.runtime.max_points_per_pillar;
  runtime.min_pillars = package.runtime.min_pillars;
  runtime.max_pillars = package.runtime.max_pillars;
  runtime.max_points = package.runtime.max_points;
  runtime.frame_num = package.runtime.frame_num;
  runtime.num_point_features = package.model.num_point_features;
  runtime.bbox_max_num = package.runtime.bbox_max_num;
  runtime.out_size_factor = package.model.out_size_factor;
  runtime.score_threshold = package.runtime.score_threshold;
  runtime.decode_mode = package.model.decode_mode;
  runtime.bbox_code_size = package.model.bbox_code_size;
  runtime.norm_bbox = package.model.norm_bbox;
  runtime.has_velocity = package.model.has_velocity;
  runtime.task_num_classes = package.model.task_num_classes;
  runtime.label_count = static_cast<int>(package.model.label_names.size());
  return runtime;
}

PostprocessConfig PostprocessConfigFromPackage(const inference::PackageConfig& package) {
  PostprocessConfig postprocess;
  postprocess.nms_type = package.postprocess.nms_type;
  postprocess.nms_thresh = package.postprocess.nms_thresh;
  postprocess.nms_pre_maxsize = package.postprocess.nms_pre_maxsize;
  postprocess.nms_post_maxsize = package.postprocess.nms_post_maxsize;
  postprocess.final_score_threshold = package.postprocess.final_score_threshold;
  postprocess.circle_nms_min_radius = package.postprocess.circle_nms_min_radius;
  postprocess.label_score_thresh = package.postprocess.label_score_thresh;
  return postprocess;
}

template <typename T>
InferenceStatus WriteNpy(const std::filesystem::path& path,
                         const T* data,
                         size_t count,
                         const std::vector<size_t>& shape,
                         std::string* out_error) {
  std::ofstream output(path, std::ios::binary);
  if (!output.is_open()) {
    if (out_error != nullptr) {
      *out_error = "Failed to open dump file: " + path.string();
    }
    return INFERENCE_STATUS_BACKEND_ERROR;
  }

  std::ostringstream header_stream;
  header_stream << "{'descr': '" << DtypeTag<T>() << "', 'fortran_order': False, 'shape': "
                << ShapeToNpyString(shape) << ", }";
  std::string header = header_stream.str();
  const size_t base_length = 10 + header.size() + 1;
  const size_t padding = (16 - (base_length % 16)) % 16;
  header.append(padding, ' ');
  header.push_back('\n');

  const char magic[] = "\x93NUMPY";
  output.write(magic, 6);
  const char version[2] = {1, 0};
  output.write(version, 2);
  const uint16_t header_len = static_cast<uint16_t>(header.size());
  output.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
  output.write(header.data(), static_cast<std::streamsize>(header.size()));
  if (count > 0) {
    output.write(reinterpret_cast<const char*>(data),
                 static_cast<std::streamsize>(count * sizeof(T)));
  }
  return INFERENCE_STATUS_OK;
}

template <typename T>
std::string BuildNumericSummary(const std::string& name,
                                const std::string& dtype,
                                const std::vector<size_t>& shape,
                                const std::vector<T>& data) {
  std::ostringstream oss;
  oss << "    \"" << name << "\": {\n";
  oss << "      \"dtype\": \"" << dtype << "\",\n";
  oss << "      \"shape\": " << ShapeToJsonArray(shape) << ",\n";
  oss << "      \"count\": " << data.size();
  if (!data.empty()) {
    const auto [min_it, max_it] = std::minmax_element(data.begin(), data.end());
    oss << ",\n      \"min\": " << *min_it << ",\n      \"max\": " << *max_it;
    if constexpr (std::is_floating_point_v<T>) {
      const double sum = std::accumulate(data.begin(), data.end(), 0.0);
      oss << ",\n      \"mean\": " << sum / static_cast<double>(data.size());
    }
    oss << ",\n      \"preview\": [";
    const size_t preview_count = std::min<size_t>(data.size(), 8);
    for (size_t i = 0; i < preview_count; ++i) {
      if (i > 0) {
        oss << ", ";
      }
      oss << data[i];
    }
    oss << "]";
  }
  oss << "\n    }";
  return oss.str();
}

std::string Trim(const std::string& value) {
  size_t begin = 0;
  while (begin < value.size() &&
         std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
    ++begin;
  }
  size_t end = value.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
    --end;
  }
  return value.substr(begin, end - begin);
}

std::vector<std::string> SplitWhitespace(const std::string& value) {
  std::istringstream input(value);
  std::vector<std::string> tokens;
  std::string token;
  while (input >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

std::string JoinStrings(const std::vector<std::string>& values, const std::string& separator) {
  std::ostringstream oss;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      oss << separator;
    }
    oss << values[i];
  }
  return oss.str();
}

std::string ToLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

bool ParseIntStrict(const std::string& value, int* out_value) {
  try {
    size_t parsed = 0;
    const int result = std::stoi(value, &parsed);
    if (parsed != value.size()) {
      return false;
    }
    *out_value = result;
    return true;
  } catch (...) {
    return false;
  }
}

bool ParseFloatStrict(const std::string& value, float* out_value) {
  try {
    size_t parsed = 0;
    const float result = std::stof(value, &parsed);
    if (parsed != value.size()) {
      return false;
    }
    *out_value = result;
    return true;
  } catch (...) {
    return false;
  }
}

struct PcdFieldSpec {
  std::string name;
  int size = 0;
  char type = '\0';
  int count = 1;
  size_t byte_offset = 0;
  size_t scalar_offset = 0;
};

struct PcdHeader {
  std::vector<PcdFieldSpec> fields;
  std::string data_format;
  int width = 0;
  int height = 0;
  int points = 0;
  size_t point_step = 0;
  std::streampos payload_offset{};
};

std::string PcdHeaderSummary(const std::string& path, const PcdHeader& header) {
  std::vector<std::string> field_names;
  field_names.reserve(header.fields.size());
  for (const PcdFieldSpec& field : header.fields) {
    field_names.push_back(field.name);
  }
  std::ostringstream oss;
  oss << "path=" << path << ", DATA=" << (header.data_format.empty() ? "<missing>" : header.data_format)
      << ", FIELDS=" << (field_names.empty() ? "<missing>" : JoinStrings(field_names, " "))
      << ", POINTS=" << header.points << ", WIDTH=" << header.width << ", HEIGHT=" << header.height;
  return oss.str();
}

const PcdFieldSpec* FindPcdField(const PcdHeader& header, const std::string& name) {
  const auto it = std::find_if(header.fields.begin(), header.fields.end(),
                               [&name](const PcdFieldSpec& field) { return field.name == name; });
  return it == header.fields.end() ? nullptr : &(*it);
}

InferenceStatus ParsePcdHeader(std::ifstream* input,
                               const std::string& path,
                               PcdHeader* out_header,
                               std::string* out_error) {
  PcdHeader header;
  std::vector<std::string> names;
  std::vector<int> sizes;
  std::vector<char> types;
  std::vector<int> counts;

  std::string line;
  while (std::getline(*input, line)) {
    line = Trim(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::istringstream line_stream(line);
    std::string key;
    line_stream >> key;
    std::string rest;
    std::getline(line_stream, rest);
    rest = Trim(rest);

    if (key == "FIELDS") {
      names = SplitWhitespace(rest);
    } else if (key == "SIZE") {
      const std::vector<std::string> tokens = SplitWhitespace(rest);
      sizes.clear();
      for (const std::string& token : tokens) {
        int value = 0;
        if (!ParseIntStrict(token, &value)) {
          return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                          "Invalid PCD header: non-integer SIZE value '" + token +
                              "'. path=" + path);
        }
        sizes.push_back(value);
      }
    } else if (key == "TYPE") {
      const std::vector<std::string> tokens = SplitWhitespace(rest);
      types.clear();
      for (const std::string& token : tokens) {
        types.push_back(token.empty() ? '\0' : token[0]);
      }
    } else if (key == "COUNT") {
      const std::vector<std::string> tokens = SplitWhitespace(rest);
      counts.clear();
      for (const std::string& token : tokens) {
        int value = 0;
        if (!ParseIntStrict(token, &value)) {
          return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                          "Invalid PCD header: non-integer COUNT value '" + token +
                              "'. path=" + path);
        }
        counts.push_back(value);
      }
    } else if (key == "WIDTH") {
      if (!ParseIntStrict(rest, &header.width)) {
        return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                        "Invalid PCD header: non-integer WIDTH value '" + rest +
                            "'. path=" + path);
      }
    } else if (key == "HEIGHT") {
      if (!ParseIntStrict(rest, &header.height)) {
        return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                        "Invalid PCD header: non-integer HEIGHT value '" + rest +
                            "'. path=" + path);
      }
    } else if (key == "POINTS") {
      if (!ParseIntStrict(rest, &header.points)) {
        return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                        "Invalid PCD header: non-integer POINTS value '" + rest +
                            "'. path=" + path);
      }
    } else if (key == "DATA") {
      header.data_format = ToLower(rest);
      header.payload_offset = input->tellg();
      break;
    }
  }

  if (header.data_format.empty()) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Invalid PCD header: missing DATA line. path=" + path);
  }
  if (names.empty() || sizes.empty() || types.empty()) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Invalid PCD header: FIELDS, SIZE, and TYPE are required. " +
                        PcdHeaderSummary(path, header));
  }
  if (counts.empty()) {
    counts.assign(names.size(), 1);
  }
  if (names.size() != sizes.size() || names.size() != types.size() || names.size() != counts.size()) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Invalid PCD header: FIELDS/SIZE/TYPE/COUNT lengths differ. " +
                        PcdHeaderSummary(path, header));
  }
  if (header.points <= 0) {
    header.points = header.width * std::max(header.height, 1);
  }
  if (header.points <= 0) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Invalid PCD header: POINTS must be positive. " + PcdHeaderSummary(path, header));
  }

  size_t byte_offset = 0;
  size_t scalar_offset = 0;
  for (size_t i = 0; i < names.size(); ++i) {
    if (sizes[i] <= 0 || counts[i] <= 0) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "Invalid PCD header: SIZE and COUNT must be positive. " +
                          PcdHeaderSummary(path, header));
    }
    PcdFieldSpec field;
    field.name = names[i];
    field.size = sizes[i];
    field.type = types[i];
    field.count = counts[i];
    field.byte_offset = byte_offset;
    field.scalar_offset = scalar_offset;
    header.fields.push_back(field);
    byte_offset += static_cast<size_t>(field.size) * static_cast<size_t>(field.count);
    scalar_offset += static_cast<size_t>(field.count);
  }
  header.point_step = byte_offset;
  *out_header = header;
  return INFERENCE_STATUS_OK;
}

InferenceStatus ValidateFloatField(const std::string& path,
                                   const PcdHeader& header,
                                   const std::string& field_name,
                                   std::string* out_error) {
  const PcdFieldSpec* field = FindPcdField(header, field_name);
  if (field == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Unsupported PCD: missing required field '" + field_name + "'. " +
                        PcdHeaderSummary(path, header));
  }
  if (field->type != 'F' || field->size != 4 || field->count != 1) {
    std::ostringstream oss;
    oss << "Unsupported PCD: field '" << field_name
        << "' must be float32 with COUNT 1, got TYPE=" << field->type
        << ", SIZE=" << field->size << ", COUNT=" << field->count << ". "
        << PcdHeaderSummary(path, header);
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
  }
  return INFERENCE_STATUS_OK;
}

float ReadFloat32At(const std::vector<char>& buffer, size_t offset) {
  float value = 0.0f;
  std::memcpy(&value, buffer.data() + offset, sizeof(value));
  return value;
}

InferenceStatus LoadPcdPointCloudFile(const std::string& path,
                                      int num_point_features,
                                      int max_points,
                                      std::vector<float>* out_points,
                                      std::string* out_error,
                                      PointCloudLoadInfo* out_info) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Failed to open point cloud: " + path);
  }

  PcdHeader header;
  InferenceStatus status = ParsePcdHeader(&input, path, &header, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  for (const std::string& field_name : {"x", "y", "z"}) {
    status = ValidateFloatField(path, header, field_name, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
  }
  const PcdFieldSpec* intensity = FindPcdField(header, "intensity");
  if (intensity != nullptr) {
    status = ValidateFloatField(path, header, "intensity", out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
  }

  if (header.data_format != "ascii" && header.data_format != "binary") {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Unsupported PCD DATA format '" + header.data_format +
                        "'. Supported formats: ascii, binary. " + PcdHeaderSummary(path, header));
  }

  const int point_count = std::min(header.points, max_points);
  out_points->assign(static_cast<size_t>(point_count) * num_point_features, 0.0f);
  const PcdFieldSpec* x = FindPcdField(header, "x");
  const PcdFieldSpec* y = FindPcdField(header, "y");
  const PcdFieldSpec* z = FindPcdField(header, "z");

  if (header.data_format == "ascii") {
    std::string row;
    int loaded = 0;
    while (loaded < point_count && std::getline(input, row)) {
      row = Trim(row);
      if (row.empty()) {
        continue;
      }
      const std::vector<std::string> tokens = SplitWhitespace(row);
      if (tokens.size() < header.fields.back().scalar_offset +
                              static_cast<size_t>(header.fields.back().count)) {
        std::ostringstream oss;
        oss << "Failed to read ascii PCD row " << loaded << ": expected at least "
            << header.fields.back().scalar_offset + static_cast<size_t>(header.fields.back().count)
            << " values, got " << tokens.size() << ". " << PcdHeaderSummary(path, header);
        return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
      }
      float* dst = out_points->data() + static_cast<size_t>(loaded) * num_point_features;
      if (!ParseFloatStrict(tokens[x->scalar_offset], &dst[0]) ||
          !ParseFloatStrict(tokens[y->scalar_offset], &dst[1]) ||
          !ParseFloatStrict(tokens[z->scalar_offset], &dst[2])) {
        std::ostringstream oss;
        oss << "Failed to parse ascii PCD row " << loaded
            << ": x/y/z must be float32 values. " << PcdHeaderSummary(path, header);
        return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
      }
      if (intensity != nullptr && num_point_features >= 4 &&
          !ParseFloatStrict(tokens[intensity->scalar_offset], &dst[3])) {
        std::ostringstream oss;
        oss << "Failed to parse ascii PCD row " << loaded
            << ": intensity must be a float32 value. " << PcdHeaderSummary(path, header);
        return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
      }
      ++loaded;
    }
    if (loaded != point_count) {
      std::ostringstream oss;
      oss << "Failed to read ascii PCD payload: expected " << point_count
          << " points, loaded " << loaded << ". " << PcdHeaderSummary(path, header);
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
    }
  } else {
    std::vector<char> point_buffer(header.point_step);
    for (int i = 0; i < point_count; ++i) {
      input.read(point_buffer.data(), static_cast<std::streamsize>(point_buffer.size()));
      if (!input) {
        std::ostringstream oss;
        oss << "Failed to read binary PCD payload at point " << i << " of " << point_count
            << ". " << PcdHeaderSummary(path, header);
        return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
      }
      float* dst = out_points->data() + static_cast<size_t>(i) * num_point_features;
      dst[0] = ReadFloat32At(point_buffer, x->byte_offset);
      dst[1] = ReadFloat32At(point_buffer, y->byte_offset);
      dst[2] = ReadFloat32At(point_buffer, z->byte_offset);
      if (intensity != nullptr && num_point_features >= 4) {
        dst[3] = ReadFloat32At(point_buffer, intensity->byte_offset);
      }
    }
  }

  if (out_info != nullptr) {
    out_info->path = path;
    out_info->format = header.data_format;
    out_info->fields.clear();
    for (const PcdFieldSpec& field : header.fields) {
      out_info->fields.push_back(field.name);
    }
    out_info->source_points = header.points;
    out_info->loaded_points = point_count;
    out_info->num_point_features = num_point_features;
    out_info->truncated = header.points > point_count;
  }
  return INFERENCE_STATUS_OK;
}

InferenceStatus PopulateFloatBinInfo(const std::string& path,
                                     int num_point_features,
                                     size_t source_point_count,
                                     size_t point_count,
                                     int source_features,
                                     PointCloudLoadInfo* out_info) {
  if (out_info == nullptr) {
    return INFERENCE_STATUS_OK;
  }
  out_info->path = path;
  out_info->format = "float32-bin";
  out_info->fields.clear();
  for (int i = 0; i < source_features; ++i) {
    out_info->fields.push_back("f" + std::to_string(i));
  }
  out_info->source_points = static_cast<int>(source_point_count);
  out_info->loaded_points = static_cast<int>(point_count);
  out_info->num_point_features = num_point_features;
  out_info->truncated = source_point_count > point_count;
  return INFERENCE_STATUS_OK;
}

InferenceStatus LoadFloatBinFile(const std::string& path,
                                 int num_point_features,
                                 int max_points,
                                 std::vector<float>* out_points,
                                 std::string* out_error,
                                 PointCloudLoadInfo* out_info) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input.is_open()) {
    if (out_error != nullptr) {
      *out_error = "Failed to open point cloud: " + path;
    }
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  const std::streamsize file_size = input.tellg();
  if (file_size <= 0 || (file_size % static_cast<std::streamsize>(sizeof(float))) != 0) {
    if (out_error != nullptr) {
      *out_error = "Point cloud .bin file size is not aligned to float32: " + path;
    }
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  const size_t total_floats = static_cast<size_t>(file_size) / sizeof(float);
  int source_features = 0;
  if (num_point_features > 0 && total_floats % static_cast<size_t>(num_point_features) == 0) {
    source_features = num_point_features;
  } else {
    constexpr std::array<int, 3> kCommonFeatureCounts = {5, 4, 3};
    for (const int candidate : kCommonFeatureCounts) {
      if (total_floats % static_cast<size_t>(candidate) == 0) {
        source_features = candidate;
        break;
      }
    }
  }

  if (source_features <= 0) {
    if (out_error != nullptr) {
      *out_error = "Failed to infer feature count for float32 point cloud: " + path;
    }
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  input.seekg(0, std::ios::beg);
  const size_t source_point_count = total_floats / static_cast<size_t>(source_features);
  const size_t point_count = std::min(source_point_count, static_cast<size_t>(max_points));
  std::vector<float> raw_points(point_count * static_cast<size_t>(source_features), 0.0f);
  input.read(reinterpret_cast<char*>(raw_points.data()),
             static_cast<std::streamsize>(raw_points.size() * sizeof(float)));
  if (!input) {
    if (out_error != nullptr) {
      *out_error = "Failed to read float32 point cloud payload";
    }
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  out_points->assign(point_count * static_cast<size_t>(num_point_features), 0.0f);
  size_t copy_features =
      std::min(static_cast<size_t>(source_features), static_cast<size_t>(num_point_features));
  // Match InferTools PointCloudLoader(use_timestamp=False): 5D bin input is treated as XYZI
  // and the trailing feature is zero-padded instead of forwarded verbatim.
  if (source_features >= 5 && num_point_features >= 5) {
    copy_features = 4;
  }
  for (size_t i = 0; i < point_count; ++i) {
    std::copy_n(raw_points.data() + i * static_cast<size_t>(source_features), copy_features,
                out_points->data() + i * static_cast<size_t>(num_point_features));
  }
  return PopulateFloatBinInfo(path, num_point_features, source_point_count, point_count,
                              source_features, out_info);
}

}  // namespace

InferenceStatus LoadToolConfigFromModelDir(const std::string& model_dir,
                                           const std::string& pcd_path,
                                           const std::string& dump_dir,
                                           const std::string& override_config_path,
                                           int warmup_iters,
                                           int profile_iters,
                                           float bev_resolution,
                                           ToolConfig* out_config,
                                           std::string* out_error) {
  if (out_config == nullptr) {
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  out_config->model_dir = std::filesystem::path(model_dir).lexically_normal().string();
  out_config->override_config_path =
      std::filesystem::path(override_config_path).lexically_normal().string();
  out_config->pcd_path = std::filesystem::path(pcd_path).lexically_normal().string();
  out_config->dump_dir = std::filesystem::path(dump_dir).lexically_normal().string();
  out_config->warmup_iters = warmup_iters;
  out_config->profile_iters = profile_iters;
  out_config->bev_resolution = bev_resolution;

  if (out_config->model_dir.empty() || out_config->pcd_path.empty() || out_config->dump_dir.empty()) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "model_dir, pcd_path, and dump_dir must be configured");
  }
  if (warmup_iters < 0 || profile_iters <= 0) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "warmup_iters must be >= 0 and profile_iters must be > 0");
  }
  if (bev_resolution <= 0.0f) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "bev_resolution must be positive");
  }

  InferenceStatus status = inference::LidarInferenceRuntime::LoadPackageConfig(
      out_config->model_dir, &out_config->package, out_error, out_config->override_config_path);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  out_config->runtime = RuntimeConfigFromPackage(out_config->package);
  return INFERENCE_STATUS_OK;
}

InferenceStatus LoadPcdFile(const std::string& path,
                            int num_point_features,
                            int max_points,
                            std::vector<float>* out_points,
                            std::string* out_error,
                            PointCloudLoadInfo* out_info) {
  if (out_points == nullptr) {
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }
  if (num_point_features <= 0 || max_points <= 0) {
    if (out_error != nullptr) {
      *out_error = "num_point_features and max_points must be positive";
    }
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  const std::filesystem::path input_path(path);
  if (ToLower(input_path.extension().string()) == ".bin") {
    return LoadFloatBinFile(path, num_point_features, max_points, out_points, out_error, out_info);
  }
  return LoadPcdPointCloudFile(path, num_point_features, max_points, out_points, out_error, out_info);
}

InferenceStatus RunToolPostprocessNms(const CandidateDetections& candidates,
                                      const ToolConfig& config,
                                      CandidateDetections* out_result,
                                      std::string* out_error) {
  PostprocessConfig postprocess = PostprocessConfigFromPackage(config.package);
  return RunPostprocessNms(candidates, postprocess, config.runtime, out_result, out_error);
}

InferenceStatus SaveDebugArtifacts(const DebugArtifacts& debug,
                                   const ToolConfig& config,
                                   std::string* out_error) {
  const std::filesystem::path dump_dir(config.dump_dir);
  std::filesystem::create_directories(dump_dir);

  InferenceStatus status = WriteNpy<float>(
      dump_dir / "input_points.npy", debug.input_points.data(), debug.input_points.size(),
      {debug.input_points.empty() ? 0UL
                                  : debug.input_points.size() /
                                        static_cast<size_t>(config.runtime.num_point_features),
       static_cast<size_t>(config.runtime.num_point_features)},
      out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = WriteNpy<float>(dump_dir / "pillar_voxels.npy", debug.pillar_inputs.voxels.data(),
                           debug.pillar_inputs.voxels.size(),
                           {static_cast<size_t>(debug.pillar_inputs.num_pillars),
                            static_cast<size_t>(config.runtime.max_points_per_pillar),
                            static_cast<size_t>(config.runtime.num_point_features)},
                           out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = WriteNpy<int32_t>(dump_dir / "pillar_num_points.npy", debug.pillar_inputs.num_points.data(),
                             debug.pillar_inputs.num_points.size(),
                             {debug.pillar_inputs.num_points.size()}, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = WriteNpy<int32_t>(dump_dir / "pillar_coors.npy", debug.pillar_inputs.coors.data(),
                             debug.pillar_inputs.coors.size(),
                             {static_cast<size_t>(debug.pillar_inputs.num_pillars), 4}, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = WriteNpy<float>(dump_dir / "bbox_pred.npy", debug.raw_outputs.bbox_pred.data(),
                           debug.raw_outputs.bbox_pred.size(),
                           {1UL, static_cast<size_t>(debug.raw_outputs.bbox_channels),
                            static_cast<size_t>(debug.raw_outputs.height),
                            static_cast<size_t>(debug.raw_outputs.width)},
                           out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = WriteNpy<float>(dump_dir / "cls_score.npy", debug.raw_outputs.cls_score.data(),
                           debug.raw_outputs.cls_score.size(),
                           {1UL, static_cast<size_t>(debug.raw_outputs.cls_channels),
                            static_cast<size_t>(debug.raw_outputs.height),
                            static_cast<size_t>(debug.raw_outputs.width)},
                           out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = WriteNpy<float>(dump_dir / "dir_cls_pred.npy", debug.raw_outputs.dir_cls_pred.data(),
                           debug.raw_outputs.dir_cls_pred.size(),
                           {1UL, static_cast<size_t>(debug.raw_outputs.dir_channels),
                            static_cast<size_t>(debug.raw_outputs.height),
                            static_cast<size_t>(debug.raw_outputs.width)},
                           out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = WriteNpy<float>(dump_dir / "decoded_scores.npy", debug.decoded_candidates.scores.data(),
                           debug.decoded_candidates.scores.size(),
                           {debug.decoded_candidates.scores.size()}, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = WriteNpy<float>(dump_dir / "decoded_boxes.npy", debug.decoded_candidates.boxes.data(),
                           debug.decoded_candidates.boxes.size(),
                           {debug.decoded_candidates.scores.size(), static_cast<size_t>(kBoxDim)},
                           out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  std::vector<int32_t> decoded_labels(debug.decoded_candidates.labels.begin(),
                                      debug.decoded_candidates.labels.end());
  status = WriteNpy<int32_t>(dump_dir / "decoded_labels.npy", decoded_labels.data(),
                             decoded_labels.size(), {decoded_labels.size()}, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = WriteNpy<float>(dump_dir / "final_nms_scores.npy", debug.final_nms_outputs.scores.data(),
                           debug.final_nms_outputs.scores.size(),
                           {debug.final_nms_outputs.scores.size()}, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = WriteNpy<float>(dump_dir / "final_nms_boxes.npy", debug.final_nms_outputs.boxes.data(),
                           debug.final_nms_outputs.boxes.size(),
                           {debug.final_nms_outputs.scores.size(), static_cast<size_t>(kBoxDim)},
                           out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  std::vector<int32_t> final_nms_labels(debug.final_nms_outputs.labels.begin(),
                                        debug.final_nms_outputs.labels.end());
  status = WriteNpy<int32_t>(dump_dir / "final_nms_labels.npy", final_nms_labels.data(),
                             final_nms_labels.size(), {final_nms_labels.size()}, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  std::ofstream result_json(dump_dir / "result.json");
  if (!result_json.is_open()) {
    if (out_error != nullptr) {
      *out_error = "Failed to open result.json for writing";
    }
    return INFERENCE_STATUS_BACKEND_ERROR;
  }
  result_json << "{\n";
  result_json << "  \"file\": \"" << JsonEscape(config.pcd_path) << "\",\n";
  result_json << "  \"count\": " << debug.final_nms_outputs.scores.size() << ",\n";
  result_json << "  \"box_z_origin\": \"center\",\n";
  result_json << "  \"objects\": [\n";
  for (size_t i = 0; i < debug.final_nms_outputs.scores.size(); ++i) {
    const int label = debug.final_nms_outputs.labels[i];
    const size_t fallback_index = config.package.model.label_names.empty() ? 0UL
                                                                           : config.package.model
                                                                                 .label_names.size() -
                                                                                 1UL;
    const size_t label_index =
        label >= 0 && label < static_cast<int>(config.package.model.label_names.size())
            ? static_cast<size_t>(label)
            : fallback_index;
    result_json << "    {\n";
    result_json << "      \"box\": [";
    for (int dim = 0; dim < kBoxDim; ++dim) {
      if (dim > 0) {
        result_json << ", ";
      }
      result_json << debug.final_nms_outputs.boxes[i * kBoxDim + static_cast<size_t>(dim)];
    }
    result_json << "],\n";
    result_json << "      \"score\": " << debug.final_nms_outputs.scores[i] << ",\n";
    result_json << "      \"label\": " << static_cast<int>(label_index) << ",\n";
    result_json << "      \"label_name\": \""
                << JsonEscape(config.package.model.label_names[label_index]) << "\"\n";
    result_json << "    }";
    if (i + 1 < debug.final_nms_outputs.scores.size()) {
      result_json << ",";
    }
    result_json << "\n";
  }
  result_json << "  ]\n";
  result_json << "}\n";

  std::ofstream meta(dump_dir / "metadata.json");
  meta << "{\n";
  meta << "  \"timings_ms\": {\n";
  meta << "    \"pillarize\": " << debug.timings.pillarize_ms << ",\n";
  meta << "    \"h2d_prepare\": " << debug.timings.h2d_prepare_ms << ",\n";
  meta << "    \"trt_forward\": " << debug.timings.trt_forward_ms << ",\n";
  meta << "    \"d2h_collect\": " << debug.timings.d2h_collect_ms << ",\n";
  meta << "    \"decode\": " << debug.timings.decode_ms << ",\n";
  meta << "    \"output_pack\": " << debug.timings.output_pack_ms << ",\n";
  meta << "    \"total_forward\": " << debug.timings.total_forward_ms << "\n";
  meta << "  },\n";
  meta << "  \"tensors\": {\n";
  meta << BuildNumericSummary("decoded_scores", "float32",
                              {debug.decoded_candidates.scores.size()},
                              debug.decoded_candidates.scores) << ",\n";
  meta << BuildNumericSummary("decoded_labels", "int32",
                              {debug.decoded_candidates.labels.size()},
                              debug.decoded_candidates.labels) << ",\n";
  meta << BuildNumericSummary("final_nms_scores", "float32",
                              {debug.final_nms_outputs.scores.size()},
                              debug.final_nms_outputs.scores) << "\n";
  meta << "  }\n";
  meta << "}\n";
  return INFERENCE_STATUS_OK;
}

}  // namespace inference_internal
