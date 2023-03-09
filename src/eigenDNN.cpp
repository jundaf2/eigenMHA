#include "eigenDNN.h"
#include <iostream>
namespace eigenDNN{

/*
*  linear
*/

eidnnStatus_t eidnnLinearForward(eidnnHandle_t handle,
                    const Tensor<float, 3>& x,
                    const Tensor<float, 2>& w,
                    Tensor<float, 3>& y)
{

  for(int batch=0; batch<x.dimension(0); batch++){
    y.chip(0,batch) = x.chip(0,batch).contract(w, array<IndexPair<int>,1>({IndexPair<int>(1, 0)}));
  }
  return EIDNN_STATUS_SUCCESS;
}

eidnnStatus_t eidnnLinearBackward(eidnnHandle_t handle,
                     const Tensor<float, 3>& dy,
                     const Tensor<float, 3>& x,
                     const Tensor<float, 2>& w,
                     Tensor<float, 3>& dx,
                     Tensor<float, 2>& dw)
{
  for(int batch=0; batch<x.dimension(0); batch++){
    dx.chip(0,batch) += dy.chip(0,batch).contract(w, array<IndexPair<int>,1>({IndexPair<int>(1, 0)}));
    dw += dy.chip(0,batch).contract(x.chip(0,batch), array<IndexPair<int>,1>({IndexPair<int>(0, 0)}));
  }
  return EIDNN_STATUS_SUCCESS;
}
}