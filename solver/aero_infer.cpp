// aero_infer.cpp — Aero warm-start ONNX Runtime C++ binding.
//
// Loads the unified 2D/3D Aero warm-start U-Net (ONNX, opset 17,
// input  "coarse_input"  [batch,6,X,Y,Z]  -> output "fine_fields" [batch,6,X,Y,Z])
// runs inference on a coarse-field tensor, and writes the predicted fine fields.
//
// Tensor exchange uses the Aero raw format (shared with aero_raw_io.py):
//   magic 'AERO' | int32 ndim | int32 dims[ndim] | float32 data (C-order).
//
// Usage:  aero_infer <model.onnx> <input.bin> <output.bin> [--threads N]
//
// Build:  see CMakeLists.txt (links onnxruntime).

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <chrono>

namespace {

struct RawTensor {
  std::vector<int64_t> dims;
  std::vector<float> data;
};

RawTensor read_raw(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { throw std::runtime_error("cannot open input: " + path); }
  char magic[4];
  f.read(magic, 4);
  if (std::memcmp(magic, "AERO", 4) != 0) {
    throw std::runtime_error("bad magic in " + path + " (expected 'AERO')");
  }
  int32_t ndim = 0;
  f.read(reinterpret_cast<char*>(&ndim), sizeof(int32_t));
  if (ndim <= 0 || ndim > 8) throw std::runtime_error("implausible ndim");
  RawTensor t;
  t.dims.resize(ndim);
  for (int i = 0; i < ndim; ++i) {
    int32_t d = 0;
    f.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
    t.dims[i] = d;
  }
  int64_t n = std::accumulate(t.dims.begin(), t.dims.end(),
                              int64_t{1}, std::multiplies<int64_t>());
  t.data.resize(static_cast<size_t>(n));
  f.read(reinterpret_cast<char*>(t.data.data()),
         static_cast<std::streamsize>(n * sizeof(float)));
  if (!f) throw std::runtime_error("short read on " + path);
  return t;
}

void write_raw(const std::string& path, const std::vector<int64_t>& dims,
               const float* data, int64_t n) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open output: " + path);
  f.write("AERO", 4);
  int32_t ndim = static_cast<int32_t>(dims.size());
  f.write(reinterpret_cast<const char*>(&ndim), sizeof(int32_t));
  for (int64_t d : dims) {
    int32_t d32 = static_cast<int32_t>(d);
    f.write(reinterpret_cast<const char*>(&d32), sizeof(int32_t));
  }
  f.write(reinterpret_cast<const char*>(data),
          static_cast<std::streamsize>(n * sizeof(float)));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: " << argv[0]
              << " <model.onnx> <input.bin> <output.bin> [--threads N]\n";
    return 2;
  }
  const std::string model_path = argv[1];
  const std::string input_path = argv[2];
  const std::string output_path = argv[3];
  int threads = 0;  // 0 -> ORT default
  for (int i = 4; i < argc - 1; ++i) {
    if (std::string(argv[i]) == "--threads") threads = std::atoi(argv[i + 1]);
  }

  try {
    RawTensor in = read_raw(input_path);
    int64_t in_n = std::accumulate(in.dims.begin(), in.dims.end(),
                                   int64_t{1}, std::multiplies<int64_t>());

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "aero_infer");
    Ort::SessionOptions opts;
    if (threads > 0) opts.SetIntraOpNumThreads(threads);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, model_path.c_str(), opts);

    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name = session.GetInputNameAllocated(0, alloc);
    auto out_name = session.GetOutputNameAllocated(0, alloc);
    const char* in_names[] = {in_name.get()};
    const char* out_names[] = {out_name.get()};

    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        mem, in.data.data(), static_cast<size_t>(in_n),
        in.dims.data(), in.dims.size());

    auto t0 = std::chrono::high_resolution_clock::now();
    auto outputs = session.Run(Ort::RunOptions{nullptr},
                               in_names, &in_tensor, 1, out_names, 1);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    Ort::Value& out = outputs.front();
    auto info = out.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> out_dims = info.GetShape();
    int64_t out_n = info.GetElementCount();
    const float* out_data = out.GetTensorData<float>();

    write_raw(output_path, out_dims, out_data, out_n);

    std::cout << "aero_infer: model=" << model_path
              << " in=" << input_path << " out=" << output_path << "\n";
    std::cout << "  input dims  =";
    for (int64_t d : in.dims) std::cout << " " << d;
    std::cout << "\n  output dims =";
    for (int64_t d : out_dims) std::cout << " " << d;
    std::cout << "\n  inference   = " << ms << " ms  (" << out_n
              << " elements)\n";
    return 0;
  } catch (const Ort::Exception& e) {
    std::cerr << "ONNX Runtime error: " << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
