#include <Eigen/Core>
#include <Eigen/CXX11/Tensor>
#include <Eigen/AutoDiff>
#include <random>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>
#include <map>
#include <memory>

namespace eigenDNN{

using namespace Eigen;

#ifndef cudaStream_t 

typedef int* cudaStream_t; 

#endif

typedef enum {
    EIDNN_STATUS_SUCCESS                      = 0,
    EIDNN_STATUS_NOT_INITIALIZED              = 1,
    EIDNN_STATUS_ALLOC_FAILED                 = 2,
    EIDNN_STATUS_BAD_PARAM                    = 3,
    EIDNN_STATUS_INTERNAL_ERROR               = 4,
    EIDNN_STATUS_INVALID_VALUE                = 5,
    EIDNN_STATUS_ARCH_MISMATCH                = 6,
    EIDNN_STATUS_MAPPING_ERROR                = 7,
    EIDNN_STATUS_EXECUTION_FAILED             = 8,
    EIDNN_STATUS_NOT_SUPPORTED                = 9,
    EIDNN_STATUS_LICENSE_ERROR                = 10,
    EIDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11,
    EIDNN_STATUS_RUNTIME_IN_PROGRESS          = 12,
    EIDNN_STATUS_RUNTIME_FP_OVERFLOW          = 13,
} eidnnStatus_t;

class eidnnHandle_t {
 public:
    eidnnHandle_t()  = default;
    ~eidnnHandle_t() = default;

    void 	SetStream(cudaStream_t stream) {stream_ = stream;};
    cudaStream_t GetStream() const {return stream_;};

 private:
    cudaStream_t stream_ = nullptr;
};


typedef enum {
  EIDNN_DATA_FLOAT = 0,
  EIDNN_DATA_DOUBLE = 1,
  EIDNN_DATA_HALF = 2,
  EIDNN_DATA_INT8 = 3,
  EIDNN_DATA_INT32 = 4,
  EIDNN_DATA_UINT8 = 6
} eidnnDataType_t;


static const std::map<eidnnDataType_t, size_t> kUnit = {
    {EIDNN_DATA_FLOAT, sizeof(float)},
    {EIDNN_DATA_HALF, sizeof(short)}, 
    {EIDNN_DATA_INT8, sizeof(int8_t)},
    {EIDNN_DATA_UINT8, sizeof(uint8_t)},
    {EIDNN_DATA_INT32, sizeof(int32_t)},
};

typedef enum {
  EIDNN_TENSOR_NCHW = 0,   
  EIDNN_TENSOR_NHWC = 1,      
  EIDNN_TENSOR_NCHW_VECT_C = 2,
                                    
} eidnnTensorFormat_t;



/*
 *  linear
 */
eidnnStatus_t eidnnLinearForward(eidnnHandle_t handle,
                    const Tensor<float, 3>& x,
                    const Tensor<float, 2>& w,
                    Tensor<float, 3>& y);

eidnnStatus_t eidnnLinearBackward(eidnnHandle_t handle,
                     const Tensor<float, 3>& dy,
                     const Tensor<float, 3>& x,
                     const Tensor<float, 2>& w,
                     Tensor<float, 3>& dx,
                     Tensor<float, 2>& dw);

/*
 *  softmax
 */
typedef enum {
    EIDNN_SOFTMAX_FAST     = 0, 
    EIDNN_SOFTMAX_ACCURATE = 1, 
    EIDNN_SOFTMAX_LOG      = 2
} eidnnSoftmaxAlgorithm_t;

typedef enum {
    EIDNN_SOFTMAX_MODE_INSTANCE = 0, 
    EIDNN_SOFTMAX_MODE_CHANNEL  = 1 
} eidnnSoftmaxMode_t;


eidnnStatus_t eidnnSoftmaxForward(eidnnHandle_t handle,
                    eidnnSoftmaxAlgorithm_t algo,
                    eidnnSoftmaxMode_t mode,
                    const Tensor<float, 4>& x,
                    Tensor<float, 4>& y);

eidnnStatus_t eidnnSoftmaxBackward(eidnnHandle_t handle,
                     eidnnSoftmaxAlgorithm_t algo,
                     eidnnSoftmaxMode_t mode,
                     const Tensor<float, 4>& y,
                     const Tensor<float, 4>& dy,
                     Tensor<float, 4>& dx);



/*
 *  dropout
 */
using eidnnDropoutDescriptor_t = std::tuple<float, void*, size_t, unsigned long long>;

eidnnStatus_t eidnnCreateDropoutDescriptor(eidnnDropoutDescriptor_t *dropoutDesc);
eidnnStatus_t eidnnDestroyDropoutDescriptor(eidnnDropoutDescriptor_t dropoutDesc);

eidnnStatus_t eidnnDropoutGetStatesSize(eidnnHandle_t handle, size_t *sizeInBytes);

template <typename Derived>
eidnnStatus_t eidnnDropoutGetReserveSpaceSize(const TensorBase<Derived>& x, size_t *sizeInBytes);

eidnnStatus_t eidnnSetDropoutDescriptor(
    eidnnDropoutDescriptor_t    dropoutDesc,
    eidnnHandle_t               handle,
    float                       dropout,
    void                       *states,
    size_t                      stateSizeInBytes,
    unsigned long long          seed);  
eidnnStatus_t eidnnGetDropoutDescriptor(
    eidnnDropoutDescriptor_t    dropoutDesc,
    eidnnHandle_t               handle,
    float                      *dropout,
    void                       **states,
    unsigned long long         *seed);
eidnnStatus_t eidnnRestoreDropoutDescriptor(
    eidnnDropoutDescriptor_t dropoutDesc,
    eidnnHandle_t            handle,
    float                    dropout,
    void                    *states,
    size_t                   stateSizeInBytes,
    unsigned long long       seed);



eidnnStatus_t eidnnDropoutForward(
    eidnnHandle_t                       handle,
    const eidnnDropoutDescriptor_t      dropoutDesc,
    const Tensor<float, 4>         &x,
    Tensor<float, 4>               &y,
    void                               *reserveSpace,
    size_t                              reserveSpaceSizeInBytes);

eidnnStatus_t eidnnDropoutBackward(
    eidnnHandle_t                   handle,
    const eidnnDropoutDescriptor_t  dropoutDesc,
    const Tensor<float, 4>       &dy,
    Tensor<float, 4>             &dx,
    void                           *reserveSpace,
    size_t                          reserveSpaceSizeInBytes);
}
