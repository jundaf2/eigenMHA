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
  for(int b=0; b<x.dimension(0); b++){
    y.chip(b,0) = x.chip(b,0).contract(w, array<IndexPair<int>,1>({IndexPair<int>(1, 0)}));
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
  for(int b=0; b<x.dimension(0); b++){
    dx.chip(b,0) += dy.chip(b,0).contract(w, array<IndexPair<int>,1>({IndexPair<int>(1, 0)}));
    dw += dy.chip(b,0).contract(x.chip(b,0), array<IndexPair<int>,1>({IndexPair<int>(0, 0)}));
  }
  return EIDNN_STATUS_SUCCESS;
}


eidnnStatus_t eidnnSoftmaxForward(eidnnHandle_t handle,
                    eidnnSoftmaxAlgorithm_t algo,
                    eidnnSoftmaxMode_t mode,
                    const Tensor<float, 4>& x,
                    Tensor<float, 4>& y)
{
  auto exp_max = (x - x.maximum(array<Index, 1>({3})).broadcast(array<Index, 4>({1,1,x.dimension(3)})).reshape(x.dimensions())).exp();
  y =  exp_max / exp_max.sum(array<Index, 1>({3})).broadcast(array<Index, 4>({1,1,x.dimension(3)})).reshape(x.dimensions());
  return EIDNN_STATUS_SUCCESS;
}

eidnnStatus_t eidnnSoftmaxBackward(eidnnHandle_t handle,
                     eidnnSoftmaxAlgorithm_t algo,
                     eidnnSoftmaxMode_t mode,
                     const Tensor<float, 4>& y,
                     const Tensor<float, 4>& dy,
                     Tensor<float, 4>& dx)
{
  int s_len = y.dimension(3);
  for(int b=0; b<y.dimension(0); b++){
    for(int h=0; h<y.dimension(1); h++){
      for(int s=0; s<y.dimension(2); s++){
        const Tensor<float, 1> y_ten = y.chip(b,0).chip(h,0).chip(s,0);
        const Tensor<float, 1> dy_ten = dy.chip(b,0).chip(h,0).chip(s,0);
        Map<const VectorXf> y_vec(y_ten.data(), s_len);
        Map<const VectorXf> dy_vec(dy_ten.data(), s_len);
        MatrixXf dydx_mat = static_cast<MatrixXf>(y_vec.matrix().asDiagonal()) - static_cast<MatrixXf>(y_vec*y_vec.transpose());
        VectorXf dx_vec = dydx_mat * dy_vec;
        TensorMap<const Tensor<float, 1>> dx_ten(dx_vec.data(), s_len);
        dx.chip(b,0).chip(h,0).chip(s,0) = dx_ten;
      }
    }
  }
  return EIDNN_STATUS_SUCCESS;
}

}