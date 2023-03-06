# TransTorch -- A Toy DL Library for Transformer Models.
## Plans
* Use LightSeq with Flash-Attention to train and inference Transformer models.
  * with an application in time-series-prediction
  * using both CPU and GPU
  * with Roofline analysis

<center><img src="./figures/MHA.png" ...></center>
<center>Which part will we implement in the transformer model.</center>

## Notes
### Linear

### Layer-Norm

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

