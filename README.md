# eigenDNN -- An Open-source Toy DL Testing Framework.
To clone this repo, use
```
git clone --recursive

cd ./eigen
git checkout 3.4   # use eigen 3.4 required by EigenRand
```

## Introduction
This eigenDNN mainly focuses on providing a testing framework for libraries that train and inference Deep Neural Networks. 
* Eigen serves as the computation library that generates ground truth for GPU implementations.
* Googletest serves as the verification framwork.

Currently, it focuses on

* Forward and backward of Multi-Head Attention (MHA).
  * with a pytorch `mha.py` that illustrates the multi-head attention our eigenDNN / cuTransDNN implements
  

<center><img src="./figures/MHA.png" ...></center>
<center>Which part will we implement in the transformer model.</center>

## Notes
### Linear
```
eidnnStatus_t eidnnLinearForward(eidnnHandle_t handle,
                    const Tensor<float, 3>& x, // input data
                    const Tensor<float, 2>& w, // input weight
                    Tensor<float, 3>& y
                    );
```

```
eidnnStatus_t eidnnLinearBackward(eidnnHandle_t handle,
                     const Tensor<float, 3>& dy,
                     const Tensor<float, 3>& x,
                     const Tensor<float, 2>& w,
                     Tensor<float, 3>& dx, // gradient of input data
                     Tensor<float, 2>& dw // accumulated gradient of input weight
                     );
```

### MatMul

$$ C = \beta * C + \alpha*Op_c(MatMul(Op_a(A),Op_b(B))) $$

, where $Op_m(M)$ is whether to transpose matrix $M$ or not in the forward pass.

```
eidnnStatus_t eidnnStridedBatchGemmForward(
    eidnnHandle_t handle,
    float alpha,
    float beta,
    bool trans_A, // Op_a
    bool trans_B, // Op_b
    bool trans_C, // Op_c
    const Tensor<float, 4> &A, 
    const Tensor<float, 4> &B, 
    Tensor<float, 4> &C);
```

```
eidnnStatus_t eidnnStridedBatchGemmBackward(
    eidnnHandle_t handle,
    float alpha,
    float beta,
    bool trans_A, // Op_a or Op_b
    bool trans_d_C, // Op_c
    bool trans_d_B, // Op_b or Op_a
    const Tensor<float, 4> &A, // A or B
    const Tensor<float, 4> &d_C, // gradient of C
    Tensor<float, 4> &d_B // gradient of B or A
    );
```
### Softmax
* [cudnnSoftmaxForward()](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSoftmaxForward)
* [cudnnSoftmaxBackward()](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSoftmaxBackward)
```
eidnnStatus_t eidnnSoftmaxForward(eidnnHandle_t handle,
                    eidnnSoftmaxAlgorithm_t algo,
                    eidnnSoftmaxMode_t mode,
                    const Tensor<float, 4>& x,
                    Tensor<float, 4>& y);
```

```
eidnnStatus_t eidnnSoftmaxBackward(eidnnHandle_t handle,
                     eidnnSoftmaxAlgorithm_t algo,
                     eidnnSoftmaxMode_t mode,
                     const Tensor<float, 4>& y,
                     const Tensor<float, 4>& dy,
                     Tensor<float, 4>& dx);
```

### Dropout
* [cudnnCreateDropoutDescriptor()]()
* [cudnnDestroyDropoutDescriptor()]()
* [cudnnDropoutGetStatesSize()]()
* [cudnnDropoutGetReserveSpaceSize()]()
* [cudnnDropoutForward()]()
* [cudnnGetDropoutDescriptor()]()
* [cudnnRestoreDropoutDescriptor()]()
* [cudnnSetDropoutDescriptor()]()
* [cudnnDropoutBackward()]()
```
// dropout rate, 
// pointer to memory space of states (allocated by forward pass), 
// size of memory space in bytes (calculated by forward pass), 
// random seed
using eidnnDropoutDescriptor_t = std::tuple<float, void*, size_t, unsigned long long>; 
```
```
eidnnStatus_t eidnnDropoutForward(
    eidnnHandle_t                       handle,
    eidnnDropoutDescriptor_t      &dropoutDesc,
    const Tensor<float, 4>         &x, // input data
    Tensor<float, 4>               &y // input data after dropout
    );
```

```
eidnnStatus_t eidnnDropoutBackward(
    eidnnHandle_t                   handle,
    const eidnnDropoutDescriptor_t  dropoutDesc,
    const Tensor<float, 4>       &dy, // gradient of dropout output data
    Tensor<float, 4>             &dx // gradient of dropout input data
    );
```

### Multi-head Attention
* [cudnnCreateAttnDescriptor()]()
* [cudnnSetAttnDescriptor()]()
* [cudnnGetAttnDescriptor()]()
* [cudnnSetAttnDescriptor()]()
* [cudnnDestroyAttnDescriptor()]()
* [cudnnGetMultiHeadAttnBuffers()]()
* [cudnnGetMultiHeadAttnWeights()]()
* [cudnnMultiHeadAttnForward()]()
* [cudnnMultiHeadAttnBackwardData()]()
* [cudnnMultiHeadAttnBackwardWeights()]()

