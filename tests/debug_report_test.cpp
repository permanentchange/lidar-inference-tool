#include <cassert>
#include <sstream>
#include "debug_report.hpp"
#include "debug_profile.hpp"
int main() {
  using namespace inference_tools;
  auto s=ComputeStats({4,1,3,2}); assert(s.n==4 && s.mean==2.5 && s.p50==2.5 && std::abs(s.p90-3.7)<1e-12);
  assert(ComputeStats({7}).p90==7);
  int calls=0, measured=0; bool stage=false;
  RunProfilePass(2,3,[&] { assert(!stage); return ++calls; },[&](int value) { assert(value>2); ++measured; });
  assert(calls==5 && measured==3); stage=true;
  RunProfilePass(2,3,[&] { assert(stage); return ++calls; },[&](int value) { assert(value>7); ++measured; });
  assert(calls==10 && measured==6);
  calls=0; RunProfilePass(0,1,[&] { return ++calls; },[](int value) { assert(value==1); }); assert(calls==1);
  std::ostringstream out; out<<Json(Json::Object{{"path","a\n\"\\b"},{"timestamp",uint64_t(1776582550193446516ULL)}});
  assert(out.str().find("a\\u000a\\\"\\\\b")!=std::string::npos);
  assert(out.str().find("1776582550193446516")!=std::string::npos);
}
