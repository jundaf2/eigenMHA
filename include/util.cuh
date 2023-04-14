#include <iostream>
#include <fstream>
#include "eigenDNN.h"
#include <cuda.h>
#include <cudnn.h>


void checkCudaError(cudaError_t code, const char *expr, const char *file, int line);
void checkCudnnError(cudnnStatus_t code, const char *expr, const char *file, int line);
bool compareResults(const float *res, const float *ref, int len);
void print_vec(const float *outv, std::string outn, int start, int end);

#define CHECK_CUDA_ERR(...)                                             \
    do {                                                                \
        checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__);  \
    } while (0)

#define CHECK_CUDNN_ERR(...)                                            \
    do {                                                                \
        checkCudnnError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
    } while (0)



void launch_mse_loss_kernel(const float* output, const float* target, float* loss, float* d_loss, int num_elem);

std::vector<float> vector0213(std::vector<float> data, int A, int B, int C, int D);

std::vector<float> vector2013(std::vector<float> data, int A, int B, int C, int D);

std::vector<float> vector0132(std::vector<float> data, int A, int B, int C, int D);

std::vector<float> vector3210(std::vector<float> data, int A, int B, int C, int D);

std::vector<float> vector01(std::vector<float> data, int A, int B);