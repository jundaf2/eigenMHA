#include <iostream>
#include <fstream>
#include "eigenDNN.h"
#include "torch/torch.h"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "nn_test.h"

using namespace std;




struct test_MHA : public nn_test::nnTest, torch::nn::Module {

  test_MHA(int batch_size, int n_heads, int seq_len_q, int seq_len_k, int head_size, float dropout_rate){
    this->hidden_size = head_size*n_heads;
    this->batch_size=batch_size;
    this->n_heads=n_heads;
    this->seq_len_q=seq_len_q;
    this->seq_len_k=seq_len_k;
    this->head_size=head_size;
    this->dropout_rate=dropout_rate;
    
    // Construct and register  submodules.
    this->q_w = register_module("q_w", torch::nn::Linear(hidden_size, hidden_size));
    this->k_w = register_module("k_w", torch::nn::Linear(hidden_size, hidden_size));
    this->v_w = register_module("v_w", torch::nn::Linear(hidden_size, hidden_size));
    this->dropout = register_module("dropout", torch::nn::Dropout(dropout_rate));
    this->softmax = register_module("softmax", torch::nn::Softmax(3));
    this->o_w = register_module("o_w", torch::nn::Linear(hidden_size, hidden_size));
  }

public:
  void init_data() override {
    
    size_t weight_len = hidden_size*hidden_size;
    size_t bias_len = hidden_size;
    size_t in_data_len_q = batch_size*seq_len_q*hidden_size;
    size_t in_data_len_k = batch_size*seq_len_k*hidden_size;
    size_t out_data_len = in_data_len_q;
    unsigned int seed = 2023;
    float rand_range = 2;
    this->set_random_seed(seed);
    this->set_print_el_num(64);
    // weight and bias for Q
    this->set_input_vec(this->gen_rand_input(-rand_range,rand_range,weight_len).data(), weight_len, "q_weight");
    this->set_input_vec(this->gen_rand_input(-rand_range,rand_range,bias_len).data(), bias_len, "q_bias");

    // weight and bias for K
    this->set_input_vec(this->gen_rand_input(-rand_range,rand_range,weight_len).data(), weight_len, "k_weight");
    this->set_input_vec(this->gen_rand_input(-rand_range,rand_range,bias_len).data(), bias_len, "k_bias");

    // weight and bias for V
    this->set_input_vec(this->gen_rand_input(-rand_range,rand_range,weight_len).data(), weight_len, "v_weight");
    this->set_input_vec(this->gen_rand_input(-rand_range,rand_range,bias_len).data(), bias_len, "v_bias");

    // weight and bias for O
    this->set_input_vec(this->gen_rand_input(-rand_range,rand_range,weight_len).data(), weight_len, "o_weight");
    this->set_input_vec(this->gen_rand_input(-rand_range,rand_range,bias_len).data(), bias_len, "o_bias");

    // input Q
    this->set_input_vec(this->gen_rand_input(-rand_range,rand_range,in_data_len_q).data(), in_data_len_q, "q_in");
    // input K
    this->set_input_vec(this->gen_rand_input(-rand_range,rand_range,in_data_len_k).data(), in_data_len_k, "k_in");
    // input V
    this->set_input_vec(this->gen_rand_input(-rand_range,rand_range,in_data_len_k).data(), in_data_len_k, "v_in");

    // target output
    this->set_input_vec(this->gen_rand_input(-rand_range,rand_range,out_data_len).data(), out_data_len, "target");
  }

  
  void run_my_dnn() override{
    using namespace Eigen;
    eigenDNN::eidnnHandle_t handle;
    void* saved_states;

    // init from init_data
    std::vector<float> vec_q_weight = this->get_input_vec("q_weight");
    std::vector<float> vec_k_weight = this->get_input_vec("k_weight");
    std::vector<float> vec_v_weight = this->get_input_vec("v_weight");
    std::vector<float> vec_o_weight = this->get_input_vec("o_weight");

    std::vector<float> vec_q_bias = this->get_input_vec("q_bias");
    std::vector<float> vec_k_bias = this->get_input_vec("k_bias");
    std::vector<float> vec_v_bias = this->get_input_vec("v_bias");
    std::vector<float> vec_o_bias = this->get_input_vec("o_bias");

    std::vector<float> vec_q_in = this->get_input_vec("q_in");
    std::vector<float> vec_k_in = this->get_input_vec("k_in");
    std::vector<float> vec_v_in = this->get_input_vec("v_in");
    std::vector<float> vec_target = this->get_input_vec("target");

    const Eigen::Tensor<float, 2> q_weight = Eigen::TensorMap<const Eigen::Tensor<float, 2, Eigen::RowMajor>>(vec_q_weight.data(), {hidden_size, hidden_size}).swap_layout().shuffle(Eigen::array<int, 2>({1,0}));
    const Eigen::Tensor<float, 2> k_weight = Eigen::TensorMap<const Eigen::Tensor<float, 2, Eigen::RowMajor>>(vec_k_weight.data(), {hidden_size, hidden_size}).swap_layout().shuffle(Eigen::array<int, 2>({1,0}));
    const Eigen::Tensor<float, 2> v_weight = Eigen::TensorMap<const Eigen::Tensor<float, 2, Eigen::RowMajor>>(vec_v_weight.data(), {hidden_size, hidden_size}).swap_layout().shuffle(Eigen::array<int, 2>({1,0}));
    const Eigen::Tensor<float, 2> o_weight = Eigen::TensorMap<const Eigen::Tensor<float, 2, Eigen::RowMajor>>(vec_o_weight.data(), {hidden_size, hidden_size}).swap_layout().shuffle(Eigen::array<int, 2>({1,0}));

    const Eigen::Tensor<float, 1> q_bias = Eigen::TensorMap<const Eigen::Tensor<float, 1, Eigen::RowMajor>>(vec_q_bias.data(), {hidden_size}).swap_layout();
    const Eigen::Tensor<float, 1> k_bias = Eigen::TensorMap<const Eigen::Tensor<float, 1, Eigen::RowMajor>>(vec_k_bias.data(), {hidden_size}).swap_layout();
    const Eigen::Tensor<float, 1> v_bias = Eigen::TensorMap<const Eigen::Tensor<float, 1, Eigen::RowMajor>>(vec_v_bias.data(), {hidden_size}).swap_layout();
    const Eigen::Tensor<float, 1> o_bias = Eigen::TensorMap<const Eigen::Tensor<float, 1, Eigen::RowMajor>>(vec_o_bias.data(), {hidden_size}).swap_layout();
    
    const Eigen::Tensor<float, 3> q_in = Eigen::TensorMap<const Eigen::Tensor<float, 3, Eigen::RowMajor>>(vec_q_in.data(), {batch_size, seq_len_q, hidden_size}).swap_layout().shuffle(Eigen::array<int, 3>({2,1,0}));
    const Eigen::Tensor<float, 3> k_in = Eigen::TensorMap<const Eigen::Tensor<float, 3, Eigen::RowMajor>>(vec_k_in.data(), {batch_size, seq_len_k, hidden_size}).swap_layout().shuffle(Eigen::array<int, 3>({2,1,0}));
    const Eigen::Tensor<float, 3> v_in = Eigen::TensorMap<const Eigen::Tensor<float, 3, Eigen::RowMajor>>(vec_v_in.data(), {batch_size, seq_len_k, hidden_size}).swap_layout().shuffle(Eigen::array<int, 3>({2,1,0}));


    const Eigen::Tensor<float, 3> target = Eigen::TensorMap<const Eigen::Tensor<float, 3, Eigen::RowMajor>>(vec_target.data(), {batch_size, seq_len_q, hidden_size}).swap_layout().shuffle(Eigen::array<int, 3>({2,1,0}));


    // no init
    Eigen::Tensor<float, 3> q_out(batch_size, seq_len_q, hidden_size);
    Eigen::Tensor<float, 3> k_out(batch_size, seq_len_k, hidden_size);
    Eigen::Tensor<float, 3> v_out(batch_size, seq_len_k, hidden_size);
    Eigen::Tensor<float, 3> o_out(batch_size, seq_len_q, hidden_size);
    Eigen::Tensor<float, 3> o_in(batch_size, seq_len_q, hidden_size);

    
    Eigen::Tensor<float, 4> q(batch_size, n_heads, seq_len_q, head_size); 
    Eigen::Tensor<float, 4> k(batch_size, n_heads, seq_len_k, head_size);
    Eigen::Tensor<float, 4> v(batch_size, n_heads, seq_len_k, head_size);
    
    Eigen::Tensor<float, 4> s(batch_size, n_heads, seq_len_q, seq_len_k);
    Eigen::Tensor<float, 4> p(batch_size, n_heads, seq_len_q, seq_len_k);
    Eigen::Tensor<float, 4> o(batch_size, n_heads, seq_len_q, head_size);
    
    Eigen::Tensor<float, 3> q_out_grad(batch_size, seq_len_q, hidden_size);
    Eigen::Tensor<float, 3> k_out_grad(batch_size, seq_len_k, hidden_size);
    Eigen::Tensor<float, 3> v_out_grad(batch_size, seq_len_k, hidden_size);
    Eigen::Tensor<float, 3> o_out_grad(batch_size, seq_len_q, hidden_size);

    Eigen::Tensor<float, 3> q_in_grad(batch_size, seq_len_q, hidden_size);
    Eigen::Tensor<float, 3> k_in_grad(batch_size, seq_len_k, hidden_size);
    Eigen::Tensor<float, 3> v_in_grad(batch_size, seq_len_k, hidden_size);
    Eigen::Tensor<float, 3> o_in_grad(batch_size, seq_len_q, hidden_size);
    
    Eigen::Tensor<float, 2> q_weight_grad(hidden_size, hidden_size); 
    Eigen::Tensor<float, 2> k_weight_grad(hidden_size, hidden_size); 
    Eigen::Tensor<float, 2> v_weight_grad(hidden_size, hidden_size); 
    Eigen::Tensor<float, 2> o_weight_grad(hidden_size, hidden_size); 

    Eigen::Tensor<float, 1> q_bias_grad(hidden_size); 
    Eigen::Tensor<float, 1> k_bias_grad(hidden_size); 
    Eigen::Tensor<float, 1> v_bias_grad(hidden_size); 
    Eigen::Tensor<float, 1> o_bias_grad(hidden_size); 
    
    Eigen::Tensor<float, 4> q_grad(batch_size, n_heads, seq_len_q, head_size);
    Eigen::Tensor<float, 4> k_grad(batch_size, n_heads, seq_len_k, head_size);
    Eigen::Tensor<float, 4> v_grad(batch_size, n_heads, seq_len_k, head_size);
    Eigen::Tensor<float, 4> s_grad(batch_size, n_heads, seq_len_q, seq_len_k); 
    Eigen::Tensor<float, 4> p_grad(batch_size, n_heads, seq_len_q, seq_len_k);
    Eigen::Tensor<float, 4> o_grad(batch_size, n_heads, seq_len_q, head_size); 
    
    Eigen::Tensor<float, 0> loss;
    Eigen::Tensor<float, 3> d_loss(batch_size, seq_len_q, hidden_size);

    // Linear Layer for Q, K and V, forward
    eigenDNN::eidnnLinearForward(handle, q_in, q_weight, q_bias, q_out);
    eigenDNN::eidnnLinearForward(handle, k_in, k_weight, k_bias, k_out);
    eigenDNN::eidnnLinearForward(handle, v_in, v_weight, v_bias, v_out);


    // reshape Q, K and V, [batch_size, seq_len, hidden_size] -> [batch_size, n_heads, seq_len, head_size]
    Eigen::Tensor<float, 3, Eigen::RowMajor> q_out_row = q_out.swap_layout().shuffle(Eigen::array<int, 3>({2,1,0}));
    Eigen::Tensor<float, 3, Eigen::RowMajor> k_out_row = k_out.swap_layout().shuffle(Eigen::array<int, 3>({2,1,0}));
    Eigen::Tensor<float, 3, Eigen::RowMajor> v_out_row = v_out.swap_layout().shuffle(Eigen::array<int, 3>({2,1,0}));
    Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>> q_0123_row(q_out_row.data(), {batch_size, seq_len_q, n_heads, head_size});
    Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>> k_0123_row(k_out_row.data(), {batch_size, seq_len_k, n_heads, head_size});
    Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>> v_0123_row(v_out_row.data(), {batch_size, seq_len_k, n_heads, head_size});
    Eigen::Tensor<float, 4> q_0123 = q_0123_row.swap_layout().shuffle(Eigen::array<int, 4>({3,2,1,0}));
    Eigen::Tensor<float, 4> k_0123 = k_0123_row.swap_layout().shuffle(Eigen::array<int, 4>({3,2,1,0}));
    Eigen::Tensor<float, 4> v_0123 = v_0123_row.swap_layout().shuffle(Eigen::array<int, 4>({3,2,1,0}));
    q = q_0123.shuffle(Eigen::array<int, 4>({0,2,1,3}));
    k = k_0123.shuffle(Eigen::array<int, 4>({0,2,1,3}));
    v = v_0123.shuffle(Eigen::array<int, 4>({0,2,1,3}));

    // S = Q*K^T, forward
    eigenDNN::eidnnStridedBatchedGemmForward(handle, 1.0f/sqrtf(head_size), 0, false, true, false, q, k, s); 

    // P = softmax(S), forward
    eigenDNN::eidnnSoftmaxForward(handle, eigenDNN::eidnnSoftmaxAlgorithm_t::EIDNN_SOFTMAX_ACCURATE, eigenDNN::eidnnSoftmaxMode_t::EIDNN_SOFTMAX_MODE_INSTANCE, s, p);

    // P = dropout(P), forward
    eigenDNN::eidnnDropoutDescriptor_t dropoutDesc = make_tuple(dropout_rate,saved_states,0,2023);
    eigenDNN::eidnnDropoutForward(handle, dropoutDesc, p, p);

    // O=P*V, forward
    eigenDNN::eidnnStridedBatchedGemmForward(handle, 1, 0, false, false, false, p, v, o);

    // reshape O, [batch_size, n_heads, seq_len, head_size] -> [batch_size, seq_len, hidden_size]
    Eigen::Tensor<float, 4> o_0213 = o.shuffle(Eigen::array<int, 4>({0,2,1,3}));
    Eigen::Tensor<float, 4, Eigen::RowMajor> o_0213_row = o_0213.swap_layout().shuffle(Eigen::array<int, 4>({3,2,1,0}));
    Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>> o_in_row(o_0213_row.data(), {batch_size, seq_len_q, hidden_size});
    o_in = o_in_row.swap_layout().shuffle(Eigen::array<int, 3>({2,1,0}));


    // Linear Layer for O, forward
    eigenDNN::eidnnLinearForward(handle, o_in, o_weight, o_bias, o_out);

    Eigen::Tensor<float, 3, Eigen::RowMajor> o_out_row = o_out.swap_layout().shuffle(Eigen::array<int, 3>({2,1,0}));
    this->register_raw_test_data(o_out_row.data(), batch_size*seq_len_q*hidden_size, "output"); 

    // MSE Loss
    eigenDNN::eidnnMSELoss(handle, o_out, target, loss, d_loss);

    // Linear Layer for O, backward
    o_out_grad = d_loss;
    eigenDNN::eidnnLinearBackward(handle, o_out_grad, o_in, o_weight, o_in_grad, o_weight_grad, o_bias_grad);

    // reshape O, [batch_size, seq_len, hidden_size] -> [batch_size, n_heads, seq_len, head_size]
    Eigen::Tensor<float, 3, Eigen::RowMajor> o_in_grad_row = o_in_grad.swap_layout().shuffle(Eigen::array<int, 3>({2,1,0}));
    Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>> o_in_grad_0123_row(o_in_grad_row.data(), {batch_size, seq_len_q, n_heads, head_size});
    Eigen::Tensor<float, 4> o_in_grad_0123 = o_in_grad_0123_row.swap_layout().shuffle(Eigen::array<int, 4>({3,2,1,0}));
    o_grad = o_in_grad_0123.shuffle(Eigen::array<int, 4>({0,2,1,3}));

    // O=P*V backward
    eigenDNN::eidnnStridedBatchedGemmBackward(handle,  1, 0, false, false, false, p, v, o_grad, p_grad, v_grad);

    // P = dropout(P), backward
    eigenDNN::eidnnDropoutBackward(handle, dropoutDesc, p_grad, p_grad);

    // P = softmax(S), backward
    eigenDNN::eidnnSoftmaxBackward(handle, eigenDNN::eidnnSoftmaxAlgorithm_t::EIDNN_SOFTMAX_ACCURATE, eigenDNN::eidnnSoftmaxMode_t::EIDNN_SOFTMAX_MODE_INSTANCE, p, p_grad, s_grad);

    // S = Q*K^T, backward
    eigenDNN::eidnnStridedBatchedGemmBackward(handle,  1.0f/sqrtf(head_size), 0, false, true, false, q, k, s_grad, q_grad, k_grad); 

    // reshape Q, K and V, [batch_size, n_heads, seq_len, head_size] -> [batch_size, seq_len, hidden_size] 
    Eigen::Tensor<float, 4> q_grad_0213 = q_grad.shuffle(Eigen::array<int, 4>({0,2,1,3}));
    Eigen::Tensor<float, 4> k_grad_0213 = k_grad.shuffle(Eigen::array<int, 4>({0,2,1,3}));
    Eigen::Tensor<float, 4> v_grad_0213 = v_grad.shuffle(Eigen::array<int, 4>({0,2,1,3}));
    Eigen::Tensor<float, 4, Eigen::RowMajor> q_grad_0213_row = q_grad_0213.swap_layout().shuffle(Eigen::array<int, 4>({3,2,1,0}));
    Eigen::Tensor<float, 4, Eigen::RowMajor> k_grad_0213_row = k_grad_0213.swap_layout().shuffle(Eigen::array<int, 4>({3,2,1,0}));
    Eigen::Tensor<float, 4, Eigen::RowMajor> v_grad_0213_row = v_grad_0213.swap_layout().shuffle(Eigen::array<int, 4>({3,2,1,0}));
    Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>> q_out_grad_row(q_grad_0213_row.data(), {batch_size, seq_len_q, hidden_size});
    Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>> k_out_grad_row(k_grad_0213_row.data(), {batch_size, seq_len_k, hidden_size});
    Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>> v_out_grad_row(v_grad_0213_row.data(), {batch_size, seq_len_k, hidden_size});
    q_out_grad = q_out_grad_row.swap_layout().shuffle(Eigen::array<int, 3>({2,1,0}));
    k_out_grad = k_out_grad_row.swap_layout().shuffle(Eigen::array<int, 3>({2,1,0}));
    v_out_grad = v_out_grad_row.swap_layout().shuffle(Eigen::array<int, 3>({2,1,0}));

    // Linear Layer for Q, K and V, backward
    eigenDNN::eidnnLinearBackward(handle, q_out_grad, q_in, q_weight, q_in_grad, q_weight_grad, q_bias_grad);
    eigenDNN::eidnnLinearBackward(handle, k_out_grad, k_in, k_weight, k_in_grad, k_weight_grad, k_bias_grad);
    eigenDNN::eidnnLinearBackward(handle, v_out_grad, v_in, v_weight, v_in_grad, v_weight_grad, v_bias_grad);

    Eigen::Tensor<float, 2, Eigen::RowMajor> q_weight_grad_row = q_weight_grad.swap_layout().shuffle(Eigen::array<int, 2>({1,0}));
    Eigen::Tensor<float, 2, Eigen::RowMajor> k_weight_grad_row = k_weight_grad.swap_layout().shuffle(Eigen::array<int, 2>({1,0}));
    Eigen::Tensor<float, 2, Eigen::RowMajor> v_weight_grad_row = v_weight_grad.swap_layout().shuffle(Eigen::array<int, 2>({1,0}));
    Eigen::Tensor<float, 2, Eigen::RowMajor> o_weight_grad_row = o_weight_grad.swap_layout().shuffle(Eigen::array<int, 2>({1,0}));

    this->register_raw_test_data(q_weight_grad_row.data(), hidden_size*hidden_size, "q_weight_grad");
    this->register_raw_test_data(k_weight_grad_row.data(), hidden_size*hidden_size, "k_weight_grad");
    this->register_raw_test_data(v_weight_grad_row.data(), hidden_size*hidden_size, "v_weight_grad");
    this->register_raw_test_data(o_weight_grad_row.data(), hidden_size*hidden_size, "o_weight_grad");

    Eigen::Tensor<float, 1, Eigen::RowMajor> q_bias_grad_row = q_bias_grad.swap_layout().shuffle(Eigen::array<int, 1>({0}));
    Eigen::Tensor<float, 1, Eigen::RowMajor> k_bias_grad_row = k_bias_grad.swap_layout().shuffle(Eigen::array<int, 1>({0}));
    Eigen::Tensor<float, 1, Eigen::RowMajor> v_bias_grad_row = v_bias_grad.swap_layout().shuffle(Eigen::array<int, 1>({0}));
    Eigen::Tensor<float, 1, Eigen::RowMajor> o_bias_grad_row = o_bias_grad.swap_layout().shuffle(Eigen::array<int, 1>({0}));

    this->register_raw_test_data(q_bias_grad_row.data(), hidden_size, "q_bias_grad");
    this->register_raw_test_data(k_bias_grad_row.data(), hidden_size, "k_bias_grad");
    this->register_raw_test_data(v_bias_grad_row.data(), hidden_size, "v_bias_grad");
    this->register_raw_test_data(o_bias_grad_row.data(), hidden_size, "o_bias_grad");

  }

  void run_torch_dnn() override{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCPU).requires_grad(true);
    // Init Input Data Tensor
    torch::Tensor Q_in = torch::empty({batch_size, seq_len_q, hidden_size}); this->get_input_ten(Q_in, "q_in", options);
    torch::Tensor K_in = torch::empty({batch_size, seq_len_k, hidden_size}); this->get_input_ten(K_in, "k_in", options); 
    torch::Tensor V_in = torch::empty({batch_size, seq_len_k, hidden_size}); this->get_input_ten(V_in, "v_in", options);
    torch::Tensor target = torch::empty({batch_size, seq_len_q, hidden_size}); this->get_input_ten(target, "target", options);

    // Init Input Weight and Bias Tensor
    this->get_input_ten(this->q_w->weight, "q_weight", options);  
    this->get_input_ten(this->k_w->weight, "k_weight", options);
    this->get_input_ten(this->v_w->weight, "v_weight", options);
    this->get_input_ten(this->o_w->weight, "o_weight", options);

    this->get_input_ten(this->q_w->bias, "q_bias", options);
    this->get_input_ten(this->k_w->bias, "k_bias", options);
    this->get_input_ten(this->v_w->bias, "v_bias", options);
    this->get_input_ten(this->o_w->bias, "o_bias", options);


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
    Q = q_w->forward(Q_in).view({batch_size, seq_len_q, n_heads, head_size}).permute({0, 2, 1, 3});
    K = k_w->forward(K_in).view({batch_size, seq_len_k, n_heads, head_size}).permute({0, 2, 1, 3});
    V = v_w->forward(V_in).view({batch_size, seq_len_k, n_heads, head_size}).permute({0, 2, 1, 3});
    S = torch::matmul(Q, K.permute({0, 1, 3, 2})) / torch::sqrt(torch::tensor(head_size)); // - (1.0 - mask.unsqueeze(1).unsqueeze(2).to(torch::kFloat32)) * 10000.0;
    P = softmax(S);
    P = dropout(P);
    O = torch::matmul(P, V).permute({0, 2, 1, 3}).contiguous().view({batch_size, seq_len_q, hidden_size});
    torch::Tensor O_out = o_w(O).view({batch_size, seq_len_q, hidden_size});
    return O_out;
  }

private:
  torch::Tensor Q, K, V, S, P, O;
  torch::nn::Linear q_w{nullptr}, k_w{nullptr}, v_w{nullptr}, o_w{nullptr};
  torch::nn::Dropout dropout{nullptr}; // (0.5, is_training())
  torch::nn::Softmax softmax{nullptr}; // (3)
  int batch_size, n_heads, seq_len_q, seq_len_k, head_size, hidden_size;
  float dropout_rate;
};

int eval_mha(unsigned batch_size,unsigned n_heads,unsigned seq_len_q,unsigned seq_len_k,unsigned head_size,float dropout_rate){
  test_MHA test_mha(batch_size,n_heads,seq_len_q,seq_len_k,head_size,dropout_rate);
  test_mha.init_data();
  test_mha.run_my_dnn();
  test_mha.run_torch_dnn();
  test_mha.verify();
}

TEST_CASE("MHA", "[mha]") {
  SECTION("[2,3,4,5,6,0.5]") {
    eval_mha(2,3,4,5,6,0);
  }
  SECTION("[4,5,6,7,8,0.5]") {
    eval_mha(4,5,6,7,8,0);
  }

  // SECTION("[2,4,32,64,0]") {
  //   eval_mha(2,4,32,64,0);
  // }

  // SECTION("[8,4,128,64,0]") {
  //   eval_mha(8,4,128,64,0);
  // }
}