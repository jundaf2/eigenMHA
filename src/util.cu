
#include "util.cuh"

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
  bool is_near2 = true;
  for (unsigned int i = 0; i < len; i++) {
    // if(!NEAR2(static_cast<float>(res[i]), ref[i], 1e-1)){
    //   std::cout << i << ": " << res[i] << " " << ref[i] << std::endl;
    // }
      is_near2 &= NEAR2(static_cast<float>(res[i]), ref[i], 1e-1);
  }
  return is_near2;
}

void checkCudaError(cudaError_t code, const char *expr, const char *file, int line) {
  if (code) {
      fprintf(stderr, "ERROR: CUDA error at %s:%d, code=%d (%s) in '%s'\n\n",
              file, line, (int)code, cudaGetErrorString(code), expr);
      exit(1);
  }
}

void checkCudnnError(cudnnStatus_t code, const char *expr, const char *file, int line) {
  if (code) {
      fprintf(stderr, "CUDNN error at %s:%d, code=%d (%s) in '%s'\n\n",
              file, line, (int)code, cudnnGetErrorString(code), expr);
      exit(1);
  }
}

/* mse loss kernel
@ target, output, d_loss are is 2d data [batch,out_features]
@ loss is a scalar
*/
__global__ void mse_loss_kernel(const float* output, const float* target, float* loss, float* d_loss, int num_elem){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(idx==0) *loss=0;

  if(idx<num_elem)
  {
      float err = output[idx] - target[idx];
      float err2 = err * err;
      float mean_square_error = err2/num_elem;
      atomicAdd(loss, mean_square_error); // poor performance
      d_loss[idx] = 2 * err * (1.0f/num_elem);
  }
}

void launch_mse_loss_kernel(const float* output, const float* target, float* loss, float* d_loss, int num_elem){
  dim3 blocks((num_elem - 1) / 512 + 1);
  dim3 threads(512);
  mse_loss_kernel<<<blocks, threads>>>(output, target, loss, d_loss, num_elem);
  CHECK_CUDA_ERR(cudaDeviceSynchronize());
}

std::vector<float> vector0213(std::vector<float> data, int A, int B, int C, int D){
  assert(data.size()==A*B*C*D);
  std::vector<float> temp_data = data;
  for(int a=0;a<A;a++)
  for(int b=0;b<B;b++)
  for(int c=0;c<C;c++)
  for(int d=0;d<D;d++){
    temp_data.at(a*(B*C*D)+(c*B+b)*D+d) = data.at(a*(B*C*D)+(b*C+c)*D+d);
  }
  return temp_data;
}

std::vector<float> vector0132(std::vector<float> data, int A, int B, int C, int D){
  assert(data.size()==A*B*C*D);
  std::vector<float> temp_data = data;
  for(int a=0;a<A;a++)
  for(int b=0;b<B;b++)
  for(int c=0;c<C;c++)
  for(int d=0;d<D;d++){
    temp_data.at(a*(B*C*D)+(b*D+d)*C+c) = data.at(a*(B*C*D)+(b*C+c)*D+d);
  }
  return temp_data;
}

std::vector<float> vector01(std::vector<float> data, int A, int B){
  assert(data.size()==A*B);
  std::vector<float> temp_data = data;
  for(int a=0;a<A;a++)
    for(int b=0;b<B;b++){
      temp_data.at(b*A+a) = data.at(a*B+b);
    }
  return temp_data;
}


std::vector<float> vector3210(std::vector<float> data, int A, int B, int C, int D){
  assert(data.size()==A*B*C*D);
  std::vector<float> temp_data = data;
  for(int a=0;a<A;a++)
  for(int b=0;b<B;b++)
  for(int c=0;c<C;c++)
  for(int d=0;d<D;d++){
    temp_data.at(d*(A*B*C)+(c*B+b)*A+a) = data.at(a*(B*C*D)+(b*C+c)*D+d);
  }
  return temp_data;
}

