#include <iostream>
#include <fstream>
#include "eigenDNN.h"

using namespace std;

void test_eidnnLinearForward(unsigned batch_size,unsigned n_heads,unsigned seq_len,unsigned head_size,unsigned hidden_size,float dropout_rate, void* saved_states){
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

void test_eidnnLinearBackward(unsigned batch_size,unsigned n_heads,unsigned seq_len,unsigned head_size,unsigned hidden_size,float dropout_rate, void* saved_states){
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


void test_eidnnSoftmaxForward(unsigned batch_size,unsigned n_heads,unsigned seq_len,unsigned head_size,unsigned hidden_size,float dropout_rate, void* saved_states){
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

void test_eidnnSoftmaxBackward(unsigned batch_size,unsigned n_heads,unsigned seq_len,unsigned head_size,unsigned hidden_size,float dropout_rate, void* saved_states){
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


void test_eidnnDropoutForwardBackward(unsigned batch_size,unsigned n_heads,unsigned seq_len,unsigned head_size,unsigned hidden_size,float dropout_rate, void* saved_states){
  using namespace Eigen;
  eigenDNN::eidnnHandle_t handle;

  Tensor<float, 4> do_in(batch_size, n_heads, seq_len, seq_len);
  Tensor<float, 4> do_out(batch_size, n_heads, seq_len, seq_len);

  do_out.setConstant(0);
  do_in.setConstant(1);

  eigenDNN::eidnnDropoutDescriptor_t dropoutDesc = make_tuple(dropout_rate,saved_states,0,2023);
  
  eigenDNN::eidnnDropoutForward(handle, dropoutDesc, do_in, do_out);
  cout << "do_in: " << endl << do_in << endl;
  cout << "do_out: " << endl << do_out << endl;


  Tensor<float, 4> do_in_grad(batch_size, n_heads, seq_len, seq_len);
  Tensor<float, 4> do_out_grad(batch_size, n_heads, seq_len, seq_len);

  do_in_grad.setConstant(0);
  do_out_grad.setConstant(1);

  eigenDNN::eidnnDropoutBackward(handle, dropoutDesc, do_out_grad, do_in_grad);
  cout << "do_out_grad: " << endl << do_out_grad << endl;
  cout << "do_in_grad: " << endl << do_in_grad << endl;
}


void test_eidnnMatMulForwardBackward(unsigned batch_size,unsigned n_heads,unsigned seq_len,unsigned head_size,unsigned hidden_size,float dropout_rate, void* saved_states){
  using namespace Eigen;
  eigenDNN::eidnnHandle_t handle;

  Tensor<float, 4> a(batch_size, n_heads, seq_len, head_size); 
  Tensor<float, 4> b(batch_size, n_heads, seq_len, head_size);
  Tensor<float, 4> c(batch_size, n_heads, seq_len, seq_len);

  
  Tensor<float, 4> a_grad(batch_size, n_heads, seq_len, head_size);
  Tensor<float, 4> b_grad(batch_size, n_heads, seq_len, head_size);
  Tensor<float, 4> c_grad(batch_size, n_heads, seq_len, seq_len); 


  c.setConstant(0);
  a.setConstant(1);
  b.setConstant(1);

  a_grad.setConstant(0);
  b_grad.setConstant(0);
  c_grad.setConstant(1);
  
  eigenDNN::eidnnStridedBatchGemmForward(handle, 1, 0, false, true, false, a, b, c);
  cout << "a: " << endl << a << endl;
  cout << "b: " << endl << b << endl;
  cout << "c: " << endl << c << endl;

  eigenDNN::eidnnStridedBatchGemmBackward(handle,  1, 0, true, false, false, b, c_grad, a_grad);
  eigenDNN::eidnnStridedBatchGemmBackward(handle,  1, 0, false, false, true, a, c_grad, b_grad);
  cout << "a_grad: " << endl << a_grad << endl;
  cout << "b_grad: " << endl << b_grad << endl;
  cout << "c_grad: " << endl << c_grad << endl;
}

void test_eidnnMSELoss(unsigned batch_size,unsigned n_heads,unsigned seq_len,unsigned head_size,unsigned hidden_size,float dropout_rate, void* saved_states){
  using namespace Eigen;
  eigenDNN::eidnnHandle_t handle;

  Tensor<float, 3> output(batch_size, seq_len, hidden_size);
  Tensor<float, 3> target(batch_size, seq_len, hidden_size);
  Tensor<float, 0> loss;
  Tensor<float, 3> d_loss(batch_size, seq_len, hidden_size);

  output.setConstant(1);
  target.setConstant(0.5);
  loss.setConstant(0);
  d_loss.setConstant(0);

  eigenDNN::eidnnMSELoss(handle, output, target, loss, d_loss);
  cout << "output: " << endl << output << endl;
  cout << "target: " << endl << target << endl;
  cout << "loss: " << endl << loss << endl;
  cout << "d_loss: " << endl << d_loss << endl;
}

int main(int argc, char* argv[]) {
  unsigned batch_size = 2;
  unsigned n_heads = 2;
  unsigned seq_len = 2;
  unsigned head_size = 2;
  unsigned hidden_size = head_size*n_heads;
  float dropout_rate = 0.1;
  void* saved_states;
  
  test_eidnnLinearForward(batch_size,n_heads,seq_len,head_size,hidden_size,dropout_rate,saved_states);
  test_eidnnLinearBackward(batch_size,n_heads,seq_len,head_size,hidden_size,dropout_rate,saved_states);
  test_eidnnSoftmaxForward(batch_size,n_heads,seq_len,head_size,hidden_size,dropout_rate,saved_states);
  test_eidnnSoftmaxBackward(batch_size,n_heads,seq_len,head_size,hidden_size,dropout_rate,saved_states);
  test_eidnnDropoutForwardBackward(batch_size,n_heads,seq_len,head_size,hidden_size,dropout_rate,saved_states);
  test_eidnnMatMulForwardBackward(batch_size,n_heads,seq_len,head_size,hidden_size,dropout_rate,saved_states);
  test_eidnnMSELoss(batch_size,n_heads,seq_len,head_size,hidden_size,dropout_rate,saved_states);
  return 0;
}
