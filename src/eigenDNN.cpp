#include "eigenDNN.h"
#include <iostream>
namespace eigenDNN{

/*
*  linear
*/

eidnnStatus_t eidnnLinearForward(eidnnHandle_t handle,
                    const Tensor<float, 3>& x,
                    const Tensor<float, 2>& w,
                    const Tensor<float, 1>& bias,
                    Tensor<float, 3>& y)
{
  for(int b=0; b<x.dimension(0); b++){
    Tensor<float, 2> xw = x.chip(b,0).contract(w, Eigen::array<IndexPair<int>,1>({IndexPair<int>(1, 0)}));
    y.chip(b,0) = xw + bias.reshape(Eigen::array<Index, 2>({1, bias.dimension(0)})).broadcast(Eigen::array<Index, 2>({xw.dimension(0),1}));
  }
  return EIDNN_STATUS_SUCCESS;
}

eidnnStatus_t eidnnLinearBackward(eidnnHandle_t handle,
                     const Tensor<float, 3>& dy,
                     const Tensor<float, 3>& x,
                     const Tensor<float, 2>& w,
                     Tensor<float, 3>& dx,
                     Tensor<float, 2>& dw,
                     Tensor<float, 1>& dbias)
{
  dx.setZero();
  dw.setZero();
  dbias.setZero();
  for(int b=0; b<x.dimension(0); b++){
    dx.chip(b,0) += dy.chip(b,0).contract(w, Eigen::array<IndexPair<int>,1>({IndexPair<int>(1, 0)}));
    dw += dy.chip(b,0).contract(x.chip(b,0), Eigen::array<IndexPair<int>,1>({IndexPair<int>(0, 0)}));
    dbias += dy.chip(b,0).sum(Eigen::array<Index, 1>({0}));
  }
  return EIDNN_STATUS_SUCCESS;
}


eidnnStatus_t eidnnSoftmaxForward(eidnnHandle_t handle,
                    eidnnSoftmaxAlgorithm_t algo,
                    eidnnSoftmaxMode_t mode,
                    const Tensor<float, 4>& x,
                    Tensor<float, 4>& y)
{
  int s_len = y.dimension(3);
  for(int b=0; b<y.dimension(0); b++){
    for(int h=0; h<y.dimension(1); h++){
      for(int s=0; s<y.dimension(2); s++){
        const Tensor<float, 1> x_ten = x.chip(b,0).chip(h,0).chip(s,0);
        const Tensor<float, 1> x_exp_max = (x_ten - x_ten.maximum().reshape(Eigen::array<Index, 1>({1})).broadcast(Eigen::array<Index, 1>({s_len}))).exp();
        const Tensor<float, 1> y_ten = x_exp_max / x_exp_max.sum().reshape(Eigen::array<Index, 1>({1})).broadcast(Eigen::array<Index, 1>({s_len}));
        y.chip(b,0).chip(h,0).chip(s,0) = y_ten;
      }
    }
  }
  // const Tensor<float, 4> exp_max = (x - x.maximum(Eigen::array<Index, 1>({3})).broadcast(Eigen::array<Index, 3>({1,1,s_len})).reshape(Eigen::array<Index, 4>({x.dimension(0),x.dimension(1),x.dimension(3),x.dimension(2)})).shuffle(Eigen::array<int, 4>({0,1,3,2}))).exp();
  // y =  exp_max / exp_max.sum(Eigen::array<Index, 1>({3})).broadcast(Eigen::array<Index, 3>({1,1,s_len})).reshape(Eigen::array<Index, 4>({x.dimension(0),x.dimension(1),x.dimension(3),x.dimension(2)})).shuffle(Eigen::array<int, 4>({0,1,3,2}));
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


eidnnStatus_t eidnnDropoutForward(
    eidnnHandle_t                       handle,
    eidnnDropoutDescriptor_t      &dropoutDesc,
    const Tensor<float, 4>         &x,
    Tensor<float, 4>               &y)
{
  float dropout_rate;
  unsigned long long seed;

  std::tie(dropout_rate,std::ignore,std::ignore,seed) = dropoutDesc;
  void* &reserveSpace = std::get<1>(dropoutDesc);
  size_t &reserveSpaceSizeInBytes = std::get<2>(dropoutDesc);

  reserveSpaceSizeInBytes = x.size()*sizeof(std::remove_const<typename std::remove_reference<decltype(x)>::type>::type::Scalar);
  reserveSpace = malloc(reserveSpaceSizeInBytes);
  
// std::cout << "reserveSpaceSizeInBytes : " << reserveSpaceSizeInBytes << std::endl;
  std::mt19937 mt1(seed);
  std::binomial_distribution<int> distribution(1, (1-dropout_rate));

  for(size_t i=0; i<x.size() ; i++) (reinterpret_cast<float *>(reserveSpace))[i] = (1.0f/(1-dropout_rate))*distribution(mt1);
  TensorMap<const Tensor<float, 4>> dropout_mask(reinterpret_cast<float *>(reserveSpace), x.dimensions());
  
  y = x*dropout_mask;

  return EIDNN_STATUS_SUCCESS;
}

eidnnStatus_t eidnnDropoutBackward(
    eidnnHandle_t                   handle,
    const eidnnDropoutDescriptor_t  dropoutDesc,
    const Tensor<float, 4>       &dy,
    Tensor<float, 4>             &dx)
{
  void *reserveSpace;
  size_t reserveSpaceSizeInBytes;
  float dropout_rate;
  unsigned long long seed;

  std::tie(dropout_rate,reserveSpace,reserveSpaceSizeInBytes,seed) = dropoutDesc;

  TensorMap<const Tensor<float, 4>> dropout_mask(reinterpret_cast<float *>(reserveSpace), dx.dimensions());
  
  dx = dy*dropout_mask;

  return EIDNN_STATUS_SUCCESS;
}

eidnnStatus_t eidnnStridedBatchGemm(
    eidnnHandle_t handle,
    float alpha,
    float beta,
    bool trans_A,
    bool trans_B,
    const Tensor<float, 4> &A, 
    const Tensor<float, 4> &B, 
    Tensor<float, 4> &C)
{
  for(int b=0; b<A.dimension(0); b++){
    for(int h=0; h<A.dimension(1); h++){
      C.chip(b,0).chip(h,0) = beta*C.chip(b,0).chip(h,0) + alpha*A.chip(b,0).chip(h,0).contract(B.chip(b,0).chip(h,0), Eigen::array<IndexPair<int>,1>({IndexPair<int>(trans_A?0:1, trans_B?1:0)}));
    }
  }
  return EIDNN_STATUS_SUCCESS;
}

eidnnStatus_t eidnnStridedBatchGemmForward(
    eidnnHandle_t handle,
    float alpha,
    float beta,
    bool trans_A,
    bool trans_B,
    bool trans_C,
    const Tensor<float, 4> &A, 
    const Tensor<float, 4> &B, 
    Tensor<float, 4> &C)
{
  if(!trans_C)
    eidnnStridedBatchGemm(handle,alpha,beta,trans_A,trans_B,A,B,C);
  else
    eidnnStridedBatchGemm(handle,alpha,beta,!trans_A,!trans_B,B,A,C);
  return EIDNN_STATUS_SUCCESS;
}

eidnnStatus_t eidnnStridedBatchGemmBackward(
    eidnnHandle_t handle,
    float alpha,
    float beta,
    bool trans_A,
    bool trans_B,
    bool trans_C,
    const Tensor<float, 4> &A, 
    const Tensor<float, 4> &B, 
    const Tensor<float, 4> &d_C,
    Tensor<float, 4> &d_A,
    Tensor<float, 4> &d_B)
{
  if(!trans_A)
    eidnnStridedBatchGemm(handle,alpha,beta,trans_C,!trans_B,d_C,B,d_A);
  else
    eidnnStridedBatchGemm(handle,alpha,beta,trans_B,!trans_C,B,d_C,d_A);

  if(!trans_B)
    eidnnStridedBatchGemm(handle,alpha,beta,!trans_A,trans_C,A,d_C,d_B);
  else
    eidnnStridedBatchGemm(handle,alpha,beta,!trans_C,trans_A,d_C,A,d_B);

  return EIDNN_STATUS_SUCCESS;
}

eidnnStatus_t eidnnMSELoss(
    eidnnHandle_t handle,
    const Tensor<float, 3> &output, 
    const Tensor<float, 3> &target,
    Tensor<float, 0> &loss,
    Tensor<float, 3> &d_loss)
{
  const Tensor<float, 3> mean_square_error = (output-target)*(output-target)*(1.0f/output.size());
  loss = mean_square_error.sum();
  d_loss = 2*(output-target)*(1.0f/output.size());
  return EIDNN_STATUS_SUCCESS;
}


}