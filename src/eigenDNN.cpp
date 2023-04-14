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
    Tensor<float, 2> wt = w.shuffle(Eigen::array<int, 2>({1,0}));
    Tensor<float, 2> x_mat = x.chip(b,0);
    Tensor<float, 2> x_max_w = x_mat.contract(wt, Eigen::array<IndexPair<int>,1>({IndexPair<int>(1, 0)}));
    y.chip(b,0) = x_max_w + bias.reshape(Eigen::array<Index, 2>({1, bias.dimension(0)})).broadcast(Eigen::array<Index, 2>({x_max_w.dimension(0),1})).eval();
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
    Tensor<float, 2> dy_mat = dy.chip(b,0);
    Tensor<float, 2> x_mat = x.chip(b,0);
    dx.chip(b,0) += (dy_mat.contract(w, Eigen::array<IndexPair<int>,1>({IndexPair<int>(1, 0)}))).eval();
    dw += (dy_mat.contract(x_mat, Eigen::array<IndexPair<int>,1>({IndexPair<int>(0, 0)}))).eval();
    dbias += (dy_mat.sum(Eigen::array<Index, 1>({0}))).eval();
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
  return EIDNN_STATUS_SUCCESS;
}

eidnnStatus_t eidnnMaskedSoftmaxForward(eidnnHandle_t handle,
                    eidnnSoftmaxAlgorithm_t algo,
                    eidnnSoftmaxMode_t mode,
                    const Tensor<float, 4>& x,
                    const Tensor<int, 1>& loWin,
                    const Tensor<int, 1>& hiWin,
                    Tensor<float, 4>& y)
{
  int s_len = y.dimension(3);
  for(int b=0; b<y.dimension(0); b++){
    for(int h=0; h<y.dimension(1); h++){
      for(int s=0; s<y.dimension(2); s++){
        const Tensor<float, 1> x_ten = x.chip(b,0).chip(h,0).chip(s,0);
        Tensor<float, 1> x_ten_masked(s_len);
        for(int i=0; i<s_len; i++)
        {
          x_ten_masked(i) = (i>=loWin(s)&&i<hiWin(s)) ? x_ten(i):-100000.f;
        }
        const Tensor<float, 1> x_exp_max = (x_ten_masked - x_ten_masked.maximum().reshape(Eigen::array<Index, 1>({1})).broadcast(Eigen::array<Index, 1>({s_len}))).exp();
        
        const Tensor<float, 1> y_ten = x_exp_max / x_exp_max.sum().reshape(Eigen::array<Index, 1>({1})).broadcast(Eigen::array<Index, 1>({s_len}));
        y.chip(b,0).chip(h,0).chip(s,0) = y_ten;
      }
    }
  }
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

  free(reserveSpace);

  return EIDNN_STATUS_SUCCESS;
}

eidnnStatus_t eidnnStridedBatchedGemm(
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
      const Tensor<float, 2> A_mat = A.slice(std::array<long,4>({b,h,0,0}), std::array<long,4>({1,1,A.dimension(2),A.dimension(3)})).reshape(std::array<long,2>({A.dimension(2),A.dimension(3)}));
      const Tensor<float, 2> B_mat = B.slice(std::array<long,4>({b,h,0,0}), std::array<long,4>({1,1,B.dimension(2),B.dimension(3)})).reshape(std::array<long,2>({B.dimension(2),B.dimension(3)}));

      Tensor<float, 2> C_mat = alpha*A_mat.contract(B_mat, Eigen::array<IndexPair<int>,1>({IndexPair<int>(trans_A?0:1, trans_B?1:0)}));
      C.chip(b,0).chip(h,0)= C_mat;
    }
  }
  return EIDNN_STATUS_SUCCESS;
}

eidnnStatus_t eidnnStridedBatchedGemmForward(
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
    eidnnStridedBatchedGemm(handle,alpha,beta,trans_A,trans_B,A,B,C);
  else
    eidnnStridedBatchedGemm(handle,alpha,beta,!trans_A,!trans_B,B,A,C);
  return EIDNN_STATUS_SUCCESS;
}

eidnnStatus_t eidnnStridedBatchedGemmBackward(
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
    eidnnStridedBatchedGemm(handle,alpha,beta,trans_C,!trans_B,d_C,B,d_A);
  else
    eidnnStridedBatchedGemm(handle,alpha,beta,trans_B,!trans_C,B,d_C,d_A);

  if(!trans_B)
    eidnnStridedBatchedGemm(handle,alpha,beta,!trans_A,trans_C,A,d_C,d_B);
  else
    eidnnStridedBatchedGemm(handle,alpha,beta,!trans_C,trans_A,d_C,A,d_B);

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

eidnnStatus_t eidnnMSELoss(
    eidnnHandle_t handle,
    const Tensor<float, 4> &output, 
    const Tensor<float, 4> &target,
    Tensor<float, 0> &loss,
    Tensor<float, 4> &d_loss)
{
  const Tensor<float, 4> mean_square_error = (output-target)*(output-target)*(1.0f/output.size());
  loss = mean_square_error.sum();
  d_loss = 2*(output-target)*(1.0f/output.size());
  return EIDNN_STATUS_SUCCESS;
}


}