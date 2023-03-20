# eigenMHA (eigenDNN) -- Multi-head Attention Inference and Training implemented by Eigen.
To clone this repo, use
```
git clone --recursive
```
As for how to install nnTest (based on LibTorch and CATCH2) for verification, see https://github.com/jundaf2/dnn-test-framework  [nnTest mainly focuses on providing a testing framework to train and inference Deep Neural Networks using YOUR OWN LIBRARY].


<center><img src="./figures/MHA.png" ...></center>
<center>Which part will we implement in the transformer model.</center>

## Introduction
 In this repo, we use Eigen3 to implement the forward and backward of Multi-head Attention in Transformer models. To be concrete, this eigenMHA (eigenDNN) does what the cuDNN does in the following APIs for MHA operations.
* [cudnnCreateAttnDescriptor()](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreateAttnDescriptor)
* [cudnnSetAttnDescriptor()](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetAttnDescriptor)
* [cudnnGetAttnDescriptor()](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetAttnDescriptor)
* [cudnnSetAttnDescriptor()](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetAttnDescriptor)
* [cudnnDestroyAttnDescriptor()](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDestroyAttnDescriptor)
* [cudnnGetMultiHeadAttnBuffers()](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetMultiHeadAttnBuffers)
* [cudnnGetMultiHeadAttnWeights()](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetMultiHeadAttnWeights)
* [cudnnMultiHeadAttnForward()](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnMultiHeadAttnForward)
* [cudnnMultiHeadAttnBackwardData()](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnMultiHeadAttnBackwardData)
* [cudnnMultiHeadAttnBackwardWeights()](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnMultiHeadAttnBackwardWeights)

## Project Structure
### What is MHA in Training Library?
1. Q K V obtained from embedding
2. Weights and bias for the linear layer of Q K V and O.
3. Gradients for the matrices, weights and intermeidate matrices

$$
\mathbf{Q} = \mathbf{Q}_{in}*\mathbf{W}_{Q}+\mathbf{b}_{Q}
$$

$$
\mathbf{K} = \mathbf{K}_{in}*\mathbf{W}_{K}+\mathbf{b}_{K}
$$

$$
\mathbf{V} = \mathbf{V}_{in}*\mathbf{W}_{V}+\mathbf{b}_{V}
$$

$$
\mathbf{S} = \mathbf{Q}*\mathbf{K}^T
$$

$$
\mathbf{P} = Softmax(Mask(\mathbf{S}))
$$

$$
\mathbf{O}=\mathbf{P}*\mathbf{V}
$$

$$
\mathbf{O}_out = \mathbf{O}*\mathbf{W}_{O}+\mathbf{b}_{O}
$$

<center><img src="./figures/attention_train.png" ...></center>




### The MHAs in this repo
1. a pytorch MHA in `mha.py`
2. a libtorch MHA in `mha.cc`
3. an eigen MHA in `mha.cc` and `./src/eigenDNN.cpp` (with headers in `./inlcude/eigenDNN.h`)
  
## Notes
### MSE Loss Function

Loss function, as the origin of DL system, is a basic component inside a DL system.

<center><img src="./figures/MSE Loss.PNG" ...></center>
<center> MSE Loss.</center>


```
eidnnStatus_t eidnnMSELoss(
    eidnnHandle_t handle,
    const Tensor<float, 3> &output, 
    const Tensor<float, 3> &target,
    Tensor<float, 0> &loss,
    Tensor<float, 3> &d_loss);
```

### Linear
cuDNN has no specific APIs for linear layer.

In eiDNN, we have

```
eidnnStatus_t eidnnLinearForward(eidnnHandle_t handle,
                    const Tensor<float, 3>& x, // data
                    const Tensor<float, 2>& w, // weight
                    const Tensor<float, 1>& bias, // bias
                    Tensor<float, 3>& y);
```

```
eidnnStatus_t eidnnLinearBackward(eidnnHandle_t handle,
                     const Tensor<float, 3>& dy,
                     const Tensor<float, 3>& x,
                     const Tensor<float, 2>& w,
                     Tensor<float, 3>& dx, // gradient of input data
                     Tensor<float, 2>& dw // accumulated gradient of input weight
                     );
eidnnStatus_t eidnnLinearBackward(eidnnHandle_t handle,
                     const Tensor<float, 3>& dy,
                     const Tensor<float, 3>& x,
                     const Tensor<float, 2>& w,
                     Tensor<float, 3>& dx, // gradient of input data
                     Tensor<float, 2>& dw, // accumulated gradient of weight
                     Tensor<float, 1>& dbias // accumulated gradient of bias
                     );
```

### MatMul

$$ C = \beta * C + \alpha*Op_c(MatMul(Op_a(A),Op_b(B))) $$

, where $Op_m(M)$ is whether to transpose matrix $M$ or not in the forward pass.

cuDNN has no specific APIs for matrix-multiply operation.

In eiDNN, we have

```
eidnnStatus_t eidnnStridedBatchedGemmForward(
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
eidnnStatus_t eidnnStridedBatchedGemmBackward(
    eidnnHandle_t handle,
    float alpha,
    float beta,
    bool trans_A, // Op_a
    bool trans_B, // Op_b
    bool trans_C, // Op_c
    const Tensor<float, 4> &A, // A
    const Tensor<float, 4> &B, // B
    const Tensor<float, 4> &d_C, // gradient of C
    Tensor<float, 4> &d_A, // gradient of A
    Tensor<float, 4> &d_B // gradient of B
    );
```
### Softmax
cuDNN has the following APIs for softmax operation.
* [cudnnSoftmaxForward()](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSoftmaxForward)
* [cudnnSoftmaxBackward()](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSoftmaxBackward)

In eiDNN, we have

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
cuDNN has the following APIs for dropout operation.
* [cudnnCreateDropoutDescriptor()]()
* [cudnnDestroyDropoutDescriptor()]()
* [cudnnDropoutGetStatesSize()]()
* [cudnnDropoutGetReserveSpaceSize()]()
* [cudnnDropoutForward()]()
* [cudnnGetDropoutDescriptor()]()
* [cudnnRestoreDropoutDescriptor()]()
* [cudnnSetDropoutDescriptor()]()
* [cudnnDropoutBackward()]()

In eiDNN, we have

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
