#include <iostream>
#include <fstream>
#include "eigenDNN.h"

using namespace std;

void test_eidnnLinearForward(unsigned batch_size,unsigned n_heads,unsigned seq_len,unsigned head_size,unsigned hidden_size,float dropout_rate){
  using namespace Eigen;
  eigenDNN::eidnnHandle_t handle;

  Tensor<float, 2> q_weight(hidden_size, hidden_size); 
  Tensor<float, 3> q_in(batch_size, seq_len, hidden_size);
  Tensor<float, 3> q_out(batch_size, seq_len, hidden_size);

  q_out.setConstant(0);
  q_weight.setConstant(1);
  q_in.setConstant(1);
  
  eigenDNN::eidnnLinearForward(handle, q_in, q_weight, q_out);
  cout << "q_out: " << endl << q_out << endl;
}

void test_eidnnLinearBackward(unsigned batch_size,unsigned n_heads,unsigned seq_len,unsigned head_size,unsigned hidden_size,float dropout_rate){
  using namespace Eigen;
  eigenDNN::eidnnHandle_t handle;

  Tensor<float, 3> q_out_grad(batch_size, seq_len, hidden_size);
  Tensor<float, 3> q_in(batch_size, seq_len, hidden_size);
  Tensor<float, 2> q_weight(hidden_size, hidden_size); 

  Tensor<float, 3> q_in_grad(batch_size, seq_len, hidden_size);
  Tensor<float, 2> q_weight_grad(hidden_size, hidden_size); 

  q_in_grad.setConstant(0);
  q_weight_grad.setConstant(0);

  q_out_grad.setConstant(1);
  q_in.setConstant(1);
  q_weight.setConstant(1);
  
  eigenDNN::eidnnLinearBackward(handle, q_out_grad, q_in, q_weight, q_in_grad, q_weight_grad);
  cout << "q_in_grad: " << endl << q_in_grad << endl;
  cout << "q_weight_grad: " << endl << q_weight_grad << endl;
}


void test_eidnnSoftmaxForward(unsigned batch_size,unsigned n_heads,unsigned seq_len,unsigned head_size,unsigned hidden_size,float dropout_rate){
  using namespace Eigen;
  eigenDNN::eidnnHandle_t handle;

  Tensor<float, 4> s_in(batch_size, n_heads, seq_len, seq_len);
  Tensor<float, 4> s_out(batch_size, n_heads, seq_len, seq_len);

  s_out.setConstant(0);
  s_in.setConstant(1);
  
  eigenDNN::eidnnSoftmaxForward(handle, eigenDNN::eidnnSoftmaxAlgorithm_t::EIDNN_SOFTMAX_ACCURATE, eigenDNN::eidnnSoftmaxMode_t::EIDNN_SOFTMAX_MODE_INSTANCE, s_in, s_out);
  cout << "s_in: " << endl << s_in << endl;
  cout << "s_out: " << endl << s_out << endl;
}

void test_eidnnSoftmaxBackward(unsigned batch_size,unsigned n_heads,unsigned seq_len,unsigned head_size,unsigned hidden_size,float dropout_rate){
  using namespace Eigen;
  eigenDNN::eidnnHandle_t handle;

  Tensor<float, 4> s_in_grad(batch_size, n_heads, seq_len, seq_len);
  Tensor<float, 4> s_out(batch_size, n_heads, seq_len, seq_len);
  Tensor<float, 4> s_out_grad(batch_size, n_heads, seq_len, seq_len);

  s_in_grad.setConstant(0);

  s_out_grad.setConstant(0.1);
  s_out.setConstant(1);
  
  eigenDNN::eidnnSoftmaxBackward(handle, eigenDNN::eidnnSoftmaxAlgorithm_t::EIDNN_SOFTMAX_ACCURATE, eigenDNN::eidnnSoftmaxMode_t::EIDNN_SOFTMAX_MODE_INSTANCE, s_out, s_out_grad, s_in_grad);
  cout << "s_out_grad: " << endl << s_out_grad << endl;
  cout << "s_out: " << endl << s_out << endl;
  cout << "s_in_grad: " << endl << s_in_grad << endl;
  
}

int main(int argc, char* argv[]) {
  unsigned batch_size = 2;
  unsigned n_heads = 2;
  unsigned seq_len = 2;
  unsigned head_size = 2;
  unsigned hidden_size = head_size*n_heads;
  float dropout_rate = 0.1;
  
  test_eidnnLinearForward(batch_size,n_heads,seq_len,head_size,hidden_size,dropout_rate);
  test_eidnnLinearBackward(batch_size,n_heads,seq_len,head_size,hidden_size,dropout_rate);
  test_eidnnSoftmaxForward(batch_size,n_heads,seq_len,head_size,hidden_size,dropout_rate);
  test_eidnnSoftmaxBackward(batch_size,n_heads,seq_len,head_size,hidden_size,dropout_rate);
  return 0;
}
