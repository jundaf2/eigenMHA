# cuSimpleDNN -- An Open-source Toy DL Library with cuDNN interfaces.
To clone this repo, use
```
git clone --recursive
```

## Introduction
This repo mainly focuses on rewriting the cuDNN library to train and inference Deep Neural Networks. 
* eigenDNN: a CPU version that serves as the ground truth for GPU implementation.
* cudaDNN: a GPU version that takes advantages of LightSeq and Flash-Attention 

Currently, it focuses on

* Forward and backward of Multi-Head Attention (MHA).
  * with a pytorch `mha.py` that illustrates the multi-head attention our eigenDNN / cuTransDNN implements
  

<center><img src="./figures/MHA.png" ...></center>
<center>Which part will we implement in the transformer model.</center>

## Notes
### Linear

### MatMul

### Softmax
* [cudnnSoftmaxForward()](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSoftmaxForward)
* [cudnnSoftmaxBackward()](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSoftmaxBackward)

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

