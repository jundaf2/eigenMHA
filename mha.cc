#include <iostream>
#include <fstream>
#include "eigenDNN.h"
#include "torch/torch.h"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "nn_test.h"

using namespace std;


void test_eigen_mha(unsigned batch_size,unsigned n_heads,unsigned seq_len,unsigned head_size,unsigned hidden_size,float dropout_rate, void* saved_states){
  using namespace Eigen;
  eigenDNN::eidnnHandle_t handle;

  Eigen::Tensor<float, 2> q_weight(hidden_size, hidden_size); 
  Eigen::Tensor<float, 2> k_weight(hidden_size, hidden_size); 
  Eigen::Tensor<float, 2> v_weight(hidden_size, hidden_size); 
  Eigen::Tensor<float, 2> o_weight(hidden_size, hidden_size); 

  Eigen::Tensor<float, 1> q_bias(hidden_size); 
  Eigen::Tensor<float, 1> k_bias(hidden_size); 
  Eigen::Tensor<float, 1> v_bias(hidden_size); 
  Eigen::Tensor<float, 1> o_bias(hidden_size); 
  
  Eigen::Tensor<float, 3> q_in(batch_size, seq_len, hidden_size);
  Eigen::Tensor<float, 3> k_in(batch_size, seq_len, hidden_size);
  Eigen::Tensor<float, 3> v_in(batch_size, seq_len, hidden_size);
  Eigen::Tensor<float, 3> o_in(batch_size, seq_len, hidden_size);
  
  Eigen::Tensor<float, 3> q_out(batch_size, seq_len, hidden_size);
  Eigen::Tensor<float, 3> k_out(batch_size, seq_len, hidden_size);
  Eigen::Tensor<float, 3> v_out(batch_size, seq_len, hidden_size);
  Eigen::Tensor<float, 3> o_out(batch_size, seq_len, hidden_size);
  
  Eigen::Tensor<float, 4> q(batch_size, n_heads, seq_len, head_size); 
  Eigen::Tensor<float, 4> k(batch_size, n_heads, seq_len, head_size);
  Eigen::Tensor<float, 4> v(batch_size, n_heads, seq_len, head_size);
  
  Eigen::Tensor<float, 4> s(batch_size, n_heads, seq_len, seq_len);
  Eigen::Tensor<float, 4> p(batch_size, n_heads, seq_len, seq_len);
  Eigen::Tensor<float, 4> o(batch_size, n_heads, seq_len, head_size);
  
  Eigen::Tensor<float, 3> q_out_grad(batch_size, seq_len, hidden_size);
  Eigen::Tensor<float, 3> k_out_grad(batch_size, seq_len, hidden_size);
  Eigen::Tensor<float, 3> v_out_grad(batch_size, seq_len, hidden_size);
  Eigen::Tensor<float, 3> o_out_grad(batch_size, seq_len, hidden_size);

  Eigen::Tensor<float, 3> q_in_grad(batch_size, seq_len, hidden_size);
  Eigen::Tensor<float, 3> k_in_grad(batch_size, seq_len, hidden_size);
  Eigen::Tensor<float, 3> v_in_grad(batch_size, seq_len, hidden_size);
  Eigen::Tensor<float, 3> o_in_grad(batch_size, seq_len, hidden_size);
  
  Eigen::Tensor<float, 2> q_weight_grad(hidden_size, hidden_size); 
  Eigen::Tensor<float, 2> k_weight_grad(hidden_size, hidden_size); 
  Eigen::Tensor<float, 2> v_weight_grad(hidden_size, hidden_size); 
  Eigen::Tensor<float, 2> o_weight_grad(hidden_size, hidden_size); 

  Eigen::Tensor<float, 1> q_bias_grad(hidden_size); 
  Eigen::Tensor<float, 1> k_bias_grad(hidden_size); 
  Eigen::Tensor<float, 1> v_bias_grad(hidden_size); 
  Eigen::Tensor<float, 1> o_bias_grad(hidden_size); 
  
  Eigen::Tensor<float, 4> q_grad(batch_size, n_heads, seq_len, head_size);
  Eigen::Tensor<float, 4> k_grad(batch_size, n_heads, seq_len, head_size);
  Eigen::Tensor<float, 4> v_grad(batch_size, n_heads, seq_len, head_size);
  Eigen::Tensor<float, 4> s_grad(batch_size, n_heads, seq_len, seq_len); 
  Eigen::Tensor<float, 4> p_grad(batch_size, n_heads, seq_len, seq_len);
  Eigen::Tensor<float, 4> o_grad(batch_size, n_heads, seq_len, head_size); 
  
  Eigen::Tensor<float, 3> target(batch_size, seq_len, hidden_size);
  Eigen::Tensor<float, 0> loss;
  Eigen::Tensor<float, 3> d_loss(batch_size, seq_len, hidden_size);

  q_weight.setRandom();
  k_weight.setRandom();
  v_weight.setRandom();
  o_weight.setRandom();

  q_bias.setRandom();
  k_bias.setRandom();
  v_bias.setRandom();
  o_bias.setRandom();

  q_in.setConstant(1);
  k_in.setConstant(1);
  v_in.setConstant(1);

  q_in.setRandom();
  k_in.setRandom();
  v_in.setRandom();

  target.setConstant(0.5);
  
  
  // Linear Layer for Q, K and V, forward
  eigenDNN::eidnnLinearForward(handle, q_in, q_weight, q_bias, q_out);
  eigenDNN::eidnnLinearForward(handle, k_in, k_weight, k_bias, k_out);
  eigenDNN::eidnnLinearForward(handle, v_in, v_weight, v_bias, v_out);

  // reshape Q, K and V, [batch_size, seq_len, hidden_size] -> [batch_size, n_heads, seq_len, head_size]
  Eigen::TensorMap<Eigen::Tensor<float, 4>> q_0123(q_out.data(), {batch_size, seq_len, n_heads, head_size});
  Eigen::TensorMap<Eigen::Tensor<float, 4>> k_0123(k_out.data(), {batch_size, seq_len, n_heads, head_size});
  Eigen::TensorMap<Eigen::Tensor<float, 4>> v_0123(v_out.data(), {batch_size, seq_len, n_heads, head_size});
  q = q_0123.shuffle(Eigen::array<int, 4>({0,2,1,3}));
  k = k_0123.shuffle(Eigen::array<int, 4>({0,2,1,3}));
  v = v_0123.shuffle(Eigen::array<int, 4>({0,2,1,3}));

  // S = Q*K^T, forward
  eigenDNN::eidnnStridedBatchGemmForward(handle, 1.0f/sqrtf(head_size), 0, false, true, false, q, k, s);  

  // P = softmax(S), forward
  eigenDNN::eidnnSoftmaxForward(handle, eigenDNN::eidnnSoftmaxAlgorithm_t::EIDNN_SOFTMAX_ACCURATE, eigenDNN::eidnnSoftmaxMode_t::EIDNN_SOFTMAX_MODE_INSTANCE, s, p);
  
  // P = dropout(P), forward
  eigenDNN::eidnnDropoutDescriptor_t dropoutDesc = make_tuple(dropout_rate,saved_states,0,2023);
  eigenDNN::eidnnDropoutForward(handle, dropoutDesc, p, p);
  
  // O=P*V, forward
  eigenDNN::eidnnStridedBatchGemmForward(handle, 1, 0, false, false, false, p, v, o);

  // reshape O, [batch_size, n_heads, seq_len, head_size] -> [batch_size, seq_len, hidden_size]
  o_in = Eigen::TensorMap<Eigen::Tensor<float, 3>>(static_cast<Eigen::Tensor<float, 4>>(o.shuffle(Eigen::array<int, 4>({0,2,1,3}))).data(), {batch_size, seq_len, hidden_size});
  
  // Linear Layer for O, forward
  eigenDNN::eidnnLinearForward(handle, o_in, o_weight, o_bias, o_out);
  
  // MSE Loss
  eigenDNN::eidnnMSELoss(handle, o_out, target, loss, d_loss);
  
  // Linear Layer for O, backward
  o_out_grad = d_loss;
  eigenDNN::eidnnLinearBackward(handle, o_out_grad, o_in, o_weight, o_in_grad, o_weight_grad, o_bias_grad);

  // reshape O, [batch_size, seq_len, hidden_size] -> [batch_size, n_heads, seq_len, head_size]
  Eigen::TensorMap<Eigen::Tensor<float, 4>> o_grad_0123(o_in_grad.data(), {batch_size, seq_len, n_heads, head_size});
  o_grad = o_grad_0123.shuffle(Eigen::array<int, 4>({0,2,1,3}));
  
  // O=P*V backward
  eigenDNN::eidnnStridedBatchGemmBackward(handle,  1, 0, false, false, false, p, v, o_grad, p_grad, v_grad);

  // P = dropout(P), backward
  eigenDNN::eidnnDropoutBackward(handle, dropoutDesc, p_grad, p_grad);
  
  // P = softmax(S), backward
  eigenDNN::eidnnSoftmaxBackward(handle, eigenDNN::eidnnSoftmaxAlgorithm_t::EIDNN_SOFTMAX_ACCURATE, eigenDNN::eidnnSoftmaxMode_t::EIDNN_SOFTMAX_MODE_INSTANCE, p, p_grad, s_grad);

  // S = Q*K^T, backward
  eigenDNN::eidnnStridedBatchGemmBackward(handle,  1.0f/sqrtf(head_size), 0, false, true, false, q, k, s_grad, q_grad, k_grad); 

  // reshape Q, K and V, [batch_size, n_heads, seq_len, head_size] -> [batch_size, seq_len, hidden_size] 
  q_out_grad = Eigen::TensorMap<Eigen::Tensor<float, 3>>(static_cast<Eigen::Tensor<float, 4>>(q_grad.shuffle(Eigen::array<int, 4>({0,2,1,3}))).data(), {batch_size, seq_len, hidden_size});
  k_out_grad = Eigen::TensorMap<Eigen::Tensor<float, 3>>(static_cast<Eigen::Tensor<float, 4>>(k_grad.shuffle(Eigen::array<int, 4>({0,2,1,3}))).data(), {batch_size, seq_len, hidden_size});
  v_out_grad = Eigen::TensorMap<Eigen::Tensor<float, 3>>(static_cast<Eigen::Tensor<float, 4>>(v_grad.shuffle(Eigen::array<int, 4>({0,2,1,3}))).data(), {batch_size, seq_len, hidden_size});

  // Linear Layer for Q, K and V, backward
  eigenDNN::eidnnLinearBackward(handle, q_out_grad, q_in, q_weight, q_in_grad, q_weight_grad, q_bias_grad);
  eigenDNN::eidnnLinearBackward(handle, k_out_grad, k_in, k_weight, k_in_grad, k_weight_grad, k_bias_grad);
  eigenDNN::eidnnLinearBackward(handle, v_out_grad, v_in, v_weight, v_in_grad, v_weight_grad, v_bias_grad);

  cout << "q_weight_grad: " << endl << q_weight_grad << endl;  cout << "q_bias_grad: " << endl << q_bias_grad << endl;
  cout << "k_weight_grad: " << endl << k_weight_grad << endl;  cout << "k_bias_grad: " << endl << k_bias_grad << endl;
  cout << "v_weight_grad: " << endl << v_weight_grad << endl;  cout << "v_bias_grad: " << endl << v_bias_grad << endl;
  cout << "o_weight_grad: " << endl << o_weight_grad << endl;  cout << "o_bias_grad: " << endl << o_bias_grad << endl;
}


struct MHA : torch::nn::Module {
	
  MHA(int batch_size, int n_heads, int seq_len, int head_size, int dropout_rate) {
    this->hidden_size = head_size*n_heads;
    this->batch_size=batch_size;
    this->n_heads=n_heads;
    this->seq_len=seq_len;
    this->head_size=head_size;
    this->hidden_size=hidden_size;
    this->dropout_rate=dropout_rate;
    
      // Construct and register  submodules.
    q_w = register_module("q_w", torch::nn::Linear(hidden_size, hidden_size));
    k_w = register_module("k_w", torch::nn::Linear(hidden_size, hidden_size));
    v_w = register_module("v_w", torch::nn::Linear(hidden_size, hidden_size));
    dropout = register_module("dropout", torch::nn::Dropout(dropout_rate));
    softmax = register_module("softmax", torch::nn::Softmax(-1));
    o_w = register_module("o_w", torch::nn::Linear(hidden_size, hidden_size));
  }

	// Implement the MHA's algorithm.
  torch::Tensor forward(torch::Tensor Q_in, torch::Tensor K_in, torch::Tensor V_in, torch::Tensor mask) {
    torch::Tensor Q = q_w->forward(Q_in).view({batch_size, seq_len, n_heads, head_size}).permute({0, 2, 1, 3});
    torch::Tensor K = k_w->forward(K_in).view({batch_size, seq_len, n_heads, head_size}).permute({0, 2, 1, 3});
    torch::Tensor V = v_w->forward(V_in).view({batch_size, seq_len, n_heads, head_size}).permute({0, 2, 1, 3});
    torch::Tensor S = torch::matmul(Q, K.permute({0, 1, 3, 2})) / torch::sqrt(torch::tensor(head_size)); // - (1.0 - mask.unsqueeze(1).unsqueeze(2).to(torch::kFloat32)) * 10000.0;
    torch::Tensor P = softmax(S);
    P = dropout(P);
    torch::Tensor O = torch::matmul(P, V).permute({0, 2, 1, 3}).contiguous().view({batch_size, seq_len, hidden_size});
    torch::Tensor O_out = o_w(O).view({batch_size, seq_len, hidden_size});
    return O_out;
  }

  torch::nn::Linear q_w{nullptr}, k_w{nullptr}, v_w{nullptr}, o_w{nullptr};
  torch::nn::Dropout dropout{nullptr}; // (0.5, is_training())
  torch::nn::Softmax softmax{nullptr}; // (3)
  int batch_size, n_heads, seq_len, head_size, hidden_size;
  float dropout_rate;
};

void test_torch_mha(unsigned batch_size,unsigned n_heads,unsigned seq_len,unsigned head_size,unsigned hidden_size,float dropout_rate, void* saved_states){
  auto mha = std::make_shared<MHA>(batch_size,n_heads,seq_len,head_size,dropout_rate);
  auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCPU).requires_grad(true);
  // Input Data
  torch::Tensor Q_in = torch::randn({batch_size, seq_len, hidden_size}, options);
  torch::Tensor K_in = torch::randn({batch_size, seq_len, hidden_size}, options);
  torch::Tensor V_in = torch::randn({batch_size, seq_len, hidden_size}, options);
  torch::Tensor mask = torch::randn({batch_size, seq_len}, options);  // unrelated to gradients

  torch::Tensor O_out = mha->forward(Q_in, K_in, V_in, mask);

  torch::Tensor target = 0.5*torch::ones(O_out.sizes());
  torch::Tensor loss = torch::mse_loss(target, O_out);
  loss.backward();
  cout << mha->o_w->weight << endl;
  cout << mha->o_w->bias << endl;
  cout << mha->o_w->weight.grad() << endl;
  cout << mha->o_w->bias.grad() << endl;
}

// int main(int argc, char* argv[]) {
//   unsigned batch_size = 2;
//   unsigned n_heads = 2;
//   unsigned seq_len = 2;
//   unsigned head_size = 4;
//   unsigned hidden_size = head_size*n_heads;
//   float dropout_rate = 0;
//   void* saved_states;
  
//   test_eigen_mha(batch_size,n_heads,seq_len,head_size,hidden_size,dropout_rate,saved_states);
//   test_torch_mha(batch_size,n_heads,seq_len,head_size,hidden_size,dropout_rate,saved_states);

//   return 0;
// }


struct test_MHA : public nn_test::nnTest, torch::nn::Module {

  test_MHA(int batch_size, int n_heads, int seq_len, int head_size, int dropout_rate){
    this->hidden_size = head_size*n_heads;
    this->batch_size=batch_size;
    this->n_heads=n_heads;
    this->seq_len=seq_len;
    this->head_size=head_size;
    this->hidden_size=hidden_size;
    this->dropout_rate=dropout_rate;
    
    // Construct and register  submodules.
    this->q_w = register_module("q_w", torch::nn::Linear(hidden_size, hidden_size));
    this->k_w = register_module("k_w", torch::nn::Linear(hidden_size, hidden_size));
    this->v_w = register_module("v_w", torch::nn::Linear(hidden_size, hidden_size));
    this->dropout = register_module("dropout", torch::nn::Dropout(dropout_rate));
    this->softmax = register_module("softmax", torch::nn::Softmax(-1));
    this->o_w = register_module("o_w", torch::nn::Linear(hidden_size, hidden_size));
  }

public:
  void init_data() override {
    
    size_t weight_len = hidden_size*hidden_size;
    size_t bias_len = hidden_size;
    size_t in_data_len = batch_size*seq_len*hidden_size;
    size_t out_data_len = in_data_len;
    unsigned int seed = 2023;

    // weight and bias for Q
    this->set_input_data(this->gen_input_data(weight_len, seed).data(), weight_len, "q_weight");
    this->set_input_data(this->gen_input_data(bias_len, seed).data(), bias_len, "q_bias");

    // weight and bias for K
    this->set_input_data(this->gen_input_data(weight_len, seed).data(), weight_len, "k_weight");
    this->set_input_data(this->gen_input_data(bias_len, seed).data(), bias_len, "k_bias");

    // weight and bias for V
    this->set_input_data(this->gen_input_data(weight_len, seed).data(), weight_len, "v_weight");
    this->set_input_data(this->gen_input_data(bias_len, seed).data(), bias_len, "v_bias");

    // weight and bias for O
    this->set_input_data(this->gen_input_data(weight_len, seed).data(), weight_len, "o_weight");
    this->set_input_data(this->gen_input_data(bias_len, seed).data(), bias_len, "o_bias");

    // input Q
    this->set_input_data(this->gen_input_data(in_data_len, seed).data(), in_data_len, "q_in");
    // input K
    this->set_input_data(this->gen_input_data(in_data_len, seed).data(), in_data_len, "k_in");
    // input V
    this->set_input_data(this->gen_input_data(in_data_len, seed).data(), in_data_len, "v_in");

    // target output
    this->set_input_data(this->gen_input_data(out_data_len, seed).data(), out_data_len, "target");
  }

  
  void run_my_dnn() override{
    using namespace Eigen;
    eigenDNN::eidnnHandle_t handle;
    void* saved_states;

    // init from init_data
    Eigen::TensorMap<const Eigen::Tensor<float, 2>> q_weight(this->get_input_data("q_weight").data(), {hidden_size, hidden_size}); 
    Eigen::TensorMap<const Eigen::Tensor<float, 2>> k_weight(this->get_input_data("k_weight").data(), {hidden_size, hidden_size}); 
    Eigen::TensorMap<const Eigen::Tensor<float, 2>> v_weight(this->get_input_data("v_weight").data(), {hidden_size, hidden_size}); 
    Eigen::TensorMap<const Eigen::Tensor<float, 2>> o_weight(this->get_input_data("o_weight").data(), {hidden_size, hidden_size}); 

    Eigen::TensorMap<const Eigen::Tensor<float, 1>> q_bias(this->get_input_data("q_bias").data(), {hidden_size}); 
    Eigen::TensorMap<const Eigen::Tensor<float, 1>> k_bias(this->get_input_data("k_bias").data(), {hidden_size}); 
    Eigen::TensorMap<const Eigen::Tensor<float, 1>> v_bias(this->get_input_data("v_bias").data(), {hidden_size}); 
    Eigen::TensorMap<const Eigen::Tensor<float, 1>> o_bias(this->get_input_data("o_bias").data(), {hidden_size}); 
    
    Eigen::TensorMap<const Eigen::Tensor<float, 3>> q_in(this->get_input_data("q_in").data(), {batch_size, seq_len, hidden_size});
    Eigen::TensorMap<const Eigen::Tensor<float, 3>> k_in(this->get_input_data("k_in").data(), {batch_size, seq_len, hidden_size});
    Eigen::TensorMap<const Eigen::Tensor<float, 3>> v_in(this->get_input_data("v_in").data(), {batch_size, seq_len, hidden_size});

    Eigen::TensorMap<const Eigen::Tensor<float, 3>> target(this->get_input_data("target").data(), {batch_size, seq_len, hidden_size});


    // no init
    Eigen::Tensor<float, 3> q_out(batch_size, seq_len, hidden_size);
    Eigen::Tensor<float, 3> k_out(batch_size, seq_len, hidden_size);
    Eigen::Tensor<float, 3> v_out(batch_size, seq_len, hidden_size);
    Eigen::Tensor<float, 3> o_out(batch_size, seq_len, hidden_size);
    Eigen::Tensor<float, 3> o_in(batch_size, seq_len, hidden_size);

    
    Eigen::Tensor<float, 4> q(batch_size, n_heads, seq_len, head_size); 
    Eigen::Tensor<float, 4> k(batch_size, n_heads, seq_len, head_size);
    Eigen::Tensor<float, 4> v(batch_size, n_heads, seq_len, head_size);
    
    Eigen::Tensor<float, 4> s(batch_size, n_heads, seq_len, seq_len);
    Eigen::Tensor<float, 4> p(batch_size, n_heads, seq_len, seq_len);
    Eigen::Tensor<float, 4> o(batch_size, n_heads, seq_len, head_size);
    
    Eigen::Tensor<float, 3> q_out_grad(batch_size, seq_len, hidden_size);
    Eigen::Tensor<float, 3> k_out_grad(batch_size, seq_len, hidden_size);
    Eigen::Tensor<float, 3> v_out_grad(batch_size, seq_len, hidden_size);
    Eigen::Tensor<float, 3> o_out_grad(batch_size, seq_len, hidden_size);

    Eigen::Tensor<float, 3> q_in_grad(batch_size, seq_len, hidden_size);
    Eigen::Tensor<float, 3> k_in_grad(batch_size, seq_len, hidden_size);
    Eigen::Tensor<float, 3> v_in_grad(batch_size, seq_len, hidden_size);
    Eigen::Tensor<float, 3> o_in_grad(batch_size, seq_len, hidden_size);
    
    Eigen::Tensor<float, 2> q_weight_grad(hidden_size, hidden_size); 
    Eigen::Tensor<float, 2> k_weight_grad(hidden_size, hidden_size); 
    Eigen::Tensor<float, 2> v_weight_grad(hidden_size, hidden_size); 
    Eigen::Tensor<float, 2> o_weight_grad(hidden_size, hidden_size); 

    Eigen::Tensor<float, 1> q_bias_grad(hidden_size); 
    Eigen::Tensor<float, 1> k_bias_grad(hidden_size); 
    Eigen::Tensor<float, 1> v_bias_grad(hidden_size); 
    Eigen::Tensor<float, 1> o_bias_grad(hidden_size); 
    
    Eigen::Tensor<float, 4> q_grad(batch_size, n_heads, seq_len, head_size);
    Eigen::Tensor<float, 4> k_grad(batch_size, n_heads, seq_len, head_size);
    Eigen::Tensor<float, 4> v_grad(batch_size, n_heads, seq_len, head_size);
    Eigen::Tensor<float, 4> s_grad(batch_size, n_heads, seq_len, seq_len); 
    Eigen::Tensor<float, 4> p_grad(batch_size, n_heads, seq_len, seq_len);
    Eigen::Tensor<float, 4> o_grad(batch_size, n_heads, seq_len, head_size); 
    
    Eigen::Tensor<float, 0> loss;
    Eigen::Tensor<float, 3> d_loss(batch_size, seq_len, hidden_size);

    // Linear Layer for Q, K and V, forward
    eigenDNN::eidnnLinearForward(handle, q_in, q_weight, q_bias, q_out);
    eigenDNN::eidnnLinearForward(handle, k_in, k_weight, k_bias, k_out);
    eigenDNN::eidnnLinearForward(handle, v_in, v_weight, v_bias, v_out);

    // reshape Q, K and V, [batch_size, seq_len, hidden_size] -> [batch_size, n_heads, seq_len, head_size]
    Eigen::TensorMap<Eigen::Tensor<float, 4>> q_0123(q_out.data(), {batch_size, seq_len, n_heads, head_size});
    Eigen::TensorMap<Eigen::Tensor<float, 4>> k_0123(k_out.data(), {batch_size, seq_len, n_heads, head_size});
    Eigen::TensorMap<Eigen::Tensor<float, 4>> v_0123(v_out.data(), {batch_size, seq_len, n_heads, head_size});
    q = q_0123.shuffle(Eigen::array<int, 4>({0,2,1,3}));
    k = k_0123.shuffle(Eigen::array<int, 4>({0,2,1,3}));
    v = v_0123.shuffle(Eigen::array<int, 4>({0,2,1,3}));

    // S = Q*K^T, forward
    eigenDNN::eidnnStridedBatchGemmForward(handle, 1.0f/sqrtf(head_size), 0, false, true, false, q, k, s);  

    // P = softmax(S), forward
    eigenDNN::eidnnSoftmaxForward(handle, eigenDNN::eidnnSoftmaxAlgorithm_t::EIDNN_SOFTMAX_ACCURATE, eigenDNN::eidnnSoftmaxMode_t::EIDNN_SOFTMAX_MODE_INSTANCE, s, p);
    
    // P = dropout(P), forward
    eigenDNN::eidnnDropoutDescriptor_t dropoutDesc = make_tuple(dropout_rate,saved_states,0,2023);
    eigenDNN::eidnnDropoutForward(handle, dropoutDesc, p, p);
    
    // O=P*V, forward
    eigenDNN::eidnnStridedBatchGemmForward(handle, 1, 0, false, false, false, p, v, o);

    // reshape O, [batch_size, n_heads, seq_len, head_size] -> [batch_size, seq_len, hidden_size]
    o_in = Eigen::TensorMap<Eigen::Tensor<float, 3>>(static_cast<Eigen::Tensor<float, 4>>(o.shuffle(Eigen::array<int, 4>({0,2,1,3}))).data(), {batch_size, seq_len, hidden_size});
    
    // Linear Layer for O, forward
    eigenDNN::eidnnLinearForward(handle, o_in, o_weight, o_bias, o_out);

    this->register_raw_test_data(o_out.data(), batch_size*seq_len*hidden_size, "output");
    
    // MSE Loss
    eigenDNN::eidnnMSELoss(handle, o_out, target, loss, d_loss);
    
    // Linear Layer for O, backward
    o_out_grad = d_loss;
    eigenDNN::eidnnLinearBackward(handle, o_out_grad, o_in, o_weight, o_in_grad, o_weight_grad, o_bias_grad);

    // reshape O, [batch_size, seq_len, hidden_size] -> [batch_size, n_heads, seq_len, head_size]
    Eigen::TensorMap<Eigen::Tensor<float, 4>> o_grad_0123(o_in_grad.data(), {batch_size, seq_len, n_heads, head_size});
    o_grad = o_grad_0123.shuffle(Eigen::array<int, 4>({0,2,1,3}));
    
    // O=P*V backward
    eigenDNN::eidnnStridedBatchGemmBackward(handle,  1, 0, false, false, false, p, v, o_grad, p_grad, v_grad);

    // P = dropout(P), backward
    eigenDNN::eidnnDropoutBackward(handle, dropoutDesc, p_grad, p_grad);
    
    // P = softmax(S), backward
    eigenDNN::eidnnSoftmaxBackward(handle, eigenDNN::eidnnSoftmaxAlgorithm_t::EIDNN_SOFTMAX_ACCURATE, eigenDNN::eidnnSoftmaxMode_t::EIDNN_SOFTMAX_MODE_INSTANCE, p, p_grad, s_grad);

    // S = Q*K^T, backward
    eigenDNN::eidnnStridedBatchGemmBackward(handle,  1.0f/sqrtf(head_size), 0, false, true, false, q, k, s_grad, q_grad, k_grad); 

    // reshape Q, K and V, [batch_size, n_heads, seq_len, head_size] -> [batch_size, seq_len, hidden_size] 
    q_out_grad = Eigen::TensorMap<Eigen::Tensor<float, 3>>(static_cast<Eigen::Tensor<float, 4>>(q_grad.shuffle(Eigen::array<int, 4>({0,2,1,3}))).data(), {batch_size, seq_len, hidden_size});
    k_out_grad = Eigen::TensorMap<Eigen::Tensor<float, 3>>(static_cast<Eigen::Tensor<float, 4>>(k_grad.shuffle(Eigen::array<int, 4>({0,2,1,3}))).data(), {batch_size, seq_len, hidden_size});
    v_out_grad = Eigen::TensorMap<Eigen::Tensor<float, 3>>(static_cast<Eigen::Tensor<float, 4>>(v_grad.shuffle(Eigen::array<int, 4>({0,2,1,3}))).data(), {batch_size, seq_len, hidden_size});

    // Linear Layer for Q, K and V, backward
    eigenDNN::eidnnLinearBackward(handle, q_out_grad, q_in, q_weight, q_in_grad, q_weight_grad, q_bias_grad);
    eigenDNN::eidnnLinearBackward(handle, k_out_grad, k_in, k_weight, k_in_grad, k_weight_grad, k_bias_grad);
    eigenDNN::eidnnLinearBackward(handle, v_out_grad, v_in, v_weight, v_in_grad, v_weight_grad, v_bias_grad);

    this->register_raw_test_data(q_weight_grad.data(), hidden_size*hidden_size, "q_weight_grad");
    this->register_raw_test_data(k_weight_grad.data(), hidden_size*hidden_size, "k_weight_grad");
    this->register_raw_test_data(v_weight_grad.data(), hidden_size*hidden_size, "v_weight_grad");
    this->register_raw_test_data(o_weight_grad.data(), hidden_size*hidden_size, "o_weight_grad");

    this->register_raw_test_data(q_bias_grad.data(), hidden_size, "q_bias_grad");
    this->register_raw_test_data(k_bias_grad.data(), hidden_size, "k_bias_grad");
    this->register_raw_test_data(v_bias_grad.data(), hidden_size, "v_bias_grad");
    this->register_raw_test_data(o_bias_grad.data(), hidden_size, "o_bias_grad");
  }

  void run_torch_dnn() override{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCPU).requires_grad(true);
    // Init Input Data Tensor
    torch::Tensor Q_in = this->get_input_tensor("q_in", {batch_size, seq_len, hidden_size}, options);
    torch::Tensor K_in = this->get_input_tensor("k_in", {batch_size, seq_len, hidden_size}, options);
    torch::Tensor V_in = this->get_input_tensor("v_in", {batch_size, seq_len, hidden_size}, options);
    torch::Tensor target = this->get_input_tensor("target", {batch_size, seq_len, hidden_size}, options);
    // Init Input Weight and Bias Tensor
    this->q_w->weight = this->get_input_tensor("q_weight", {hidden_size, hidden_size}, options);
    this->k_w->weight = this->get_input_tensor("k_weight", {hidden_size, hidden_size}, options);
    this->v_w->weight = this->get_input_tensor("v_weight", {hidden_size, hidden_size}, options);
    this->o_w->weight = this->get_input_tensor("o_weight", {hidden_size, hidden_size}, options);

    this->q_w->bias = this->get_input_tensor("q_bias", {hidden_size}, options);
    this->k_w->bias = this->get_input_tensor("k_bias", {hidden_size}, options);
    this->v_w->bias = this->get_input_tensor("v_bias", {hidden_size}, options);
    this->o_w->bias = this->get_input_tensor("o_bias", {hidden_size}, options);

    torch::Tensor mask; // = torch::randn({batch_size, seq_len}, options);  // unrelated to this work, we focus on gradients


    torch::Tensor O_out = this->forward(Q_in, K_in, V_in, mask);
    this->register_torch_test_data(O_out, "output");

    torch::Tensor loss = torch::mse_loss(target, O_out);
    loss.backward();
    this->register_torch_test_data(this->q_w->weight.grad(), "q_weight_grad");
    this->register_torch_test_data(this->k_w->weight.grad(), "k_weight_grad");
    this->register_torch_test_data(this->v_w->weight.grad(), "v_weight_grad");
    this->register_torch_test_data(this->o_w->weight.grad(), "o_weight_grad");

    this->register_torch_test_data(this->q_w->bias.grad(), "q_bias_grad");
    this->register_torch_test_data(this->k_w->bias.grad(), "k_bias_grad");
    this->register_torch_test_data(this->v_w->bias.grad(), "v_bias_grad");
    this->register_torch_test_data(this->o_w->bias.grad(), "o_bias_grad");
  }


  // Implement the MHA's algorithm.
  torch::Tensor forward(torch::Tensor Q_in, torch::Tensor K_in, torch::Tensor V_in, torch::Tensor mask) {
    torch::Tensor Q = q_w->forward(Q_in).view({batch_size, seq_len, n_heads, head_size}).permute({0, 2, 1, 3});
    torch::Tensor K = k_w->forward(K_in).view({batch_size, seq_len, n_heads, head_size}).permute({0, 2, 1, 3});
    torch::Tensor V = v_w->forward(V_in).view({batch_size, seq_len, n_heads, head_size}).permute({0, 2, 1, 3});
    torch::Tensor S = torch::matmul(Q, K.permute({0, 1, 3, 2})) / torch::sqrt(torch::tensor(head_size)); // - (1.0 - mask.unsqueeze(1).unsqueeze(2).to(torch::kFloat32)) * 10000.0;
    torch::Tensor P = softmax(S);
    P = dropout(P);
    torch::Tensor O = torch::matmul(P, V).permute({0, 2, 1, 3}).contiguous().view({batch_size, seq_len, hidden_size});
    torch::Tensor O_out = o_w(O).view({batch_size, seq_len, hidden_size});
    return O_out;
  }

private:
  torch::nn::Linear q_w{nullptr}, k_w{nullptr}, v_w{nullptr}, o_w{nullptr};
  torch::nn::Dropout dropout{nullptr}; // (0.5, is_training())
  torch::nn::Softmax softmax{nullptr}; // (3)
  int batch_size, n_heads, seq_len, head_size, hidden_size;
  float dropout_rate;
};

int eval_mha(unsigned batch_size,unsigned n_heads,unsigned seq_len,unsigned head_size,float dropout_rate){
  test_MHA test_mha(batch_size,n_heads,seq_len,head_size,dropout_rate);
  test_mha.init_data();
  test_mha.run_my_dnn();
  test_mha.run_torch_dnn();
  test_mha.verify();
}

TEST_CASE("MHA", "[mha]") {
  SECTION("[2,2,2,4,0.1]") {
    eval_mha(2,2,2,4,0.1);
  }
}