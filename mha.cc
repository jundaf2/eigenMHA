#include <iostream>
#include <fstream>
#include "eigenDNN.h"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using namespace std;

struct test_MHA {

  test_MHA(int batch_size, int n_heads, int seq_len_q, int seq_len_k, int head_size, float dropout_rate){
    this->hidden_size = head_size*n_heads;
    this->batch_size=batch_size;
    this->n_heads=n_heads;
    this->seq_len_q=seq_len_q;
    this->seq_len_k=seq_len_k;
    this->head_size=head_size;
    this->dropout_rate=dropout_rate;

  }

public:
  void init_data() {
    
    size_t weight_len = hidden_size*hidden_size;
    size_t bias_len = hidden_size;
    size_t in_data_len_q = batch_size*seq_len_q*hidden_size;
    size_t in_data_len_k = batch_size*seq_len_k*hidden_size;
    size_t out_data_len = in_data_len_q;
    unsigned int seed = 2023;
    float rand_range = 2;
  }

  
  void run_my_dnn(){
    using namespace Eigen;
    eigenDNN::eidnnHandle_t handle;
    void* saved_states;

    // init from init_data
    std::vector<float> vec_q_weight;
    std::vector<float> vec_k_weight;
    std::vector<float> vec_v_weight;
    std::vector<float> vec_o_weight;

    std::vector<float> vec_q_bias;
    std::vector<float> vec_k_bias;
    std::vector<float> vec_v_bias;
    std::vector<float> vec_o_bias;

    std::vector<float> vec_q_in;
    std::vector<float> vec_k_in;
    std::vector<float> vec_v_in;
    std::vector<float> vec_target;

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
    // this->register_raw_test_data(o_out_row.data(), batch_size*seq_len_q*hidden_size, "output"); 

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

    // this->register_raw_test_data(q_weight_grad_row.data(), hidden_size*hidden_size, "q_weight_grad");
    // this->register_raw_test_data(k_weight_grad_row.data(), hidden_size*hidden_size, "k_weight_grad");
    // this->register_raw_test_data(v_weight_grad_row.data(), hidden_size*hidden_size, "v_weight_grad");
    // this->register_raw_test_data(o_weight_grad_row.data(), hidden_size*hidden_size, "o_weight_grad");

    Eigen::Tensor<float, 1, Eigen::RowMajor> q_bias_grad_row = q_bias_grad.swap_layout().shuffle(Eigen::array<int, 1>({0}));
    Eigen::Tensor<float, 1, Eigen::RowMajor> k_bias_grad_row = k_bias_grad.swap_layout().shuffle(Eigen::array<int, 1>({0}));
    Eigen::Tensor<float, 1, Eigen::RowMajor> v_bias_grad_row = v_bias_grad.swap_layout().shuffle(Eigen::array<int, 1>({0}));
    Eigen::Tensor<float, 1, Eigen::RowMajor> o_bias_grad_row = o_bias_grad.swap_layout().shuffle(Eigen::array<int, 1>({0}));

    // this->register_raw_test_data(q_bias_grad_row.data(), hidden_size, "q_bias_grad");
    // this->register_raw_test_data(k_bias_grad_row.data(), hidden_size, "k_bias_grad");
    // this->register_raw_test_data(v_bias_grad_row.data(), hidden_size, "v_bias_grad");
    // this->register_raw_test_data(o_bias_grad_row.data(), hidden_size, "o_bias_grad");

  }

  void run_cudnn_dnn(){

  }



private:

  int batch_size, n_heads, seq_len_q, seq_len_k, head_size, hidden_size;
  float dropout_rate;
};

int eval_mha(unsigned batch_size,unsigned n_heads,unsigned seq_len_q,unsigned seq_len_k,unsigned head_size,float dropout_rate){
  test_MHA test_mha(batch_size,n_heads,seq_len_q,seq_len_k,head_size,dropout_rate);
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