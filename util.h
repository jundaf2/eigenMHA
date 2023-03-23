#include <iostream>
#include <fstream>
#include "eigenDNN.h"
#include <cuda.h>
#include <cudnn.h>


std::function<bool(float,float,float)> NEAR2 = [](float a, float b, float prec) -> bool { return ((a != a && b != b) 
    || (a == std::numeric_limits<typename std::remove_reference<decltype(a)>::type>::infinity() 
      && b == std::numeric_limits<typename std::remove_reference<  decltype(b)>::type>::infinity()) 
    || (-a == std::numeric_limits<typename std::remove_reference< decltype(a)>::type>::infinity() 
      && -b == std::numeric_limits<typename std::remove_reference<  decltype(b)>::type>::infinity()) 
    || (abs(a - b) / abs(a) < prec) || (abs(a - b) / abs(b) < prec) || (abs(a - b) < prec)); };

void print_vec(const float *outv, std::string outn, int start, int end) {
  std::cout << outn << ": ";
  for(int i=start; i<end; i++) {
    std::cout << outv[i] << " ";
  }
  std::cout << std::endl;
}

bool compareResults(const float *res, const float *ref, int len) {
    std::vector<float> vec_res(res,res+len);
    std::vector<float> vec_ref(ref,ref+len);
    bool is_near2 = true;
    for (unsigned int i = 0; i < len; i++) {
        is_near2 &= NEAR2(static_cast<float>(res[i]), ref[i], 1e-3);
    }
    return is_near2;
  }

inline void checkCudaError(cudaError_t code, const char *expr, const char *file, int line) {
    if (code) {
        fprintf(stderr, "ERROR: CUDA error at %s:%d, code=%d (%s) in '%s'\n\n",
                file, line, (int)code, cudaGetErrorString(code), expr);
        exit(1);
    }
}

inline void checkCudnnError(cudnnStatus_t code, const char *expr, const char *file, int line) {
    if (code) {
        fprintf(stderr, "CUDNN error at %s:%d, code=%d (%s) in '%s'\n\n",
                file, line, (int)code, cudnnGetErrorString(code), expr);
        exit(1);
    }
}

#define CHECK_CUDA_ERR(...)                                             \
    do {                                                                \
        checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__);  \
    } while (0)

#define CHECK_CUDNN_ERR(...)                                            \
    do {                                                                \
        checkCudnnError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
    } while (0)
