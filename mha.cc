#include <iostream>
#include <fstream>
#include "eigenDNN.h"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "multiHeadAttention.h"

using namespace std;

class test_MHA {
 

public:
  test_MHA(int batch_size, int n_heads, int seq_len_q, int seq_len_k, int head_size1, int head_size2, float dropout_rate, bool is_train){
    this->hidden_size1 = head_size1*n_heads;
    this->hidden_size2 = head_size2*n_heads;
    this->batch_size=batch_size;
    this->n_heads=n_heads;
    this->seq_len_q=seq_len_q;
    this->seq_len_k=seq_len_k;
    this->head_size1=head_size1;
    this->head_size2=head_size2;
    this->dropout_rate=dropout_rate;
    this->is_train = is_train;
  }

  void init_data() {
    
    size_t weight_len1 = hidden_size1*hidden_size2;
    size_t weight_len2 = hidden_size2*hidden_size2;
    size_t bias_len = hidden_size2;
    size_t in_data_len_q = batch_size*seq_len_q*hidden_size1;
    size_t in_data_len_k = batch_size*seq_len_k*hidden_size1;
    size_t out_data_len = batch_size*seq_len_q*hidden_size2;
    unsigned int seed = 2023;
    float rand_range = 2;

    vector<float> h_weight_bank(weight_len1*3+weight_len2+4*bias_len);

    h_q_in = vector<float>(in_data_len_q);
    h_k_in = vector<float>(in_data_len_k);
    h_v_in = vector<float>(in_data_len_k);
    h_target = vector<float>(out_data_len);

    // init from init_data
    const unsigned long random_seed = 2023;
    std::mt19937 generator(static_cast<unsigned int>(random_seed));
    std::uniform_real_distribution<float> uf_distribution(-1.0f, 1.0f);
    
    for (int i = 0; i < h_weight_bank.size(); i++) {
      h_weight_bank.at(i) = uf_distribution(generator); 
    }
    for (int i = 0; i < h_q_in.size(); i++) {
      h_q_in.at(i) = uf_distribution(generator); 
    }
    for (int i = 0; i < h_k_in.size(); i++) {
      h_k_in.at(i) = uf_distribution(generator); 
    }
    for (int i = 0; i < h_v_in.size(); i++) {
      h_v_in.at(i) = uf_distribution(generator); 
    }
    for (int i = 0; i < h_target.size(); i++) {
      h_target.at(i) = uf_distribution(generator); 
    }


    h_q_weight = vector<float>(h_weight_bank.begin(),h_weight_bank.begin()+in_data_len_q); 
    h_q_bias = vector<float>(h_weight_bank.begin()+weight_len1, h_weight_bank.begin()+weight_len1+bias_len);
    h_k_weight = vector<float>(h_weight_bank.begin()+weight_len1+bias_len,h_weight_bank.begin()+weight_len1*2+bias_len);
    h_k_bias = vector<float>(h_weight_bank.begin()+weight_len1*2+bias_len,h_weight_bank.begin()+weight_len1*2+bias_len*2);
    h_v_weight = vector<float>(h_weight_bank.begin()+weight_len1*2+bias_len*2,h_weight_bank.begin()+weight_len1*3+bias_len*2);
    h_v_bias = vector<float>(h_weight_bank.begin()+weight_len1*3+bias_len*2,h_weight_bank.begin()+weight_len1*3+bias_len*3);
    h_o_weight = vector<float>(h_weight_bank.begin()+weight_len1*3+bias_len*3,h_weight_bank.begin()+weight_len1*3+bias_len*3+weight_len2);
    h_o_bias = vector<float>(h_weight_bank.begin()+weight_len1*3+bias_len*3+weight_len2,h_weight_bank.begin()+weight_len1*3+bias_len*4+weight_len2);

  }

  
  void run_eigen_dnn(){
    using namespace Eigen;
    eigenDNN::eidnnHandle_t handle;
    void* saved_states;

    const Eigen::Tensor<float, 2> q_weight = Eigen::TensorMap<const Eigen::Tensor<float, 2, Eigen::RowMajor>>(h_q_weight.data(), {hidden_size2, hidden_size1}).swap_layout().shuffle(Eigen::array<int, 2>({1,0}));
    const Eigen::Tensor<float, 2> k_weight = Eigen::TensorMap<const Eigen::Tensor<float, 2, Eigen::RowMajor>>(h_k_weight.data(), {hidden_size2, hidden_size1}).swap_layout().shuffle(Eigen::array<int, 2>({1,0}));
    const Eigen::Tensor<float, 2> v_weight = Eigen::TensorMap<const Eigen::Tensor<float, 2, Eigen::RowMajor>>(h_v_weight.data(), {hidden_size2, hidden_size1}).swap_layout().shuffle(Eigen::array<int, 2>({1,0}));
    const Eigen::Tensor<float, 2> o_weight = Eigen::TensorMap<const Eigen::Tensor<float, 2, Eigen::RowMajor>>(h_o_weight.data(), {hidden_size2, hidden_size2}).swap_layout().shuffle(Eigen::array<int, 2>({1,0}));

    const Eigen::Tensor<float, 1> q_bias = Eigen::TensorMap<const Eigen::Tensor<float, 1, Eigen::RowMajor>>(h_q_bias.data(), {hidden_size2}).swap_layout();
    const Eigen::Tensor<float, 1> k_bias = Eigen::TensorMap<const Eigen::Tensor<float, 1, Eigen::RowMajor>>(h_k_bias.data(), {hidden_size2}).swap_layout();
    const Eigen::Tensor<float, 1> v_bias = Eigen::TensorMap<const Eigen::Tensor<float, 1, Eigen::RowMajor>>(h_v_bias.data(), {hidden_size2}).swap_layout();
    const Eigen::Tensor<float, 1> o_bias = Eigen::TensorMap<const Eigen::Tensor<float, 1, Eigen::RowMajor>>(h_o_bias.data(), {hidden_size2}).swap_layout();
    
    const Eigen::Tensor<float, 3> q_in = Eigen::TensorMap<const Eigen::Tensor<float, 3, Eigen::RowMajor>>(h_q_in.data(), {batch_size, seq_len_q, hidden_size1}).swap_layout().shuffle(Eigen::array<int, 3>({2,1,0}));
    const Eigen::Tensor<float, 3> k_in = Eigen::TensorMap<const Eigen::Tensor<float, 3, Eigen::RowMajor>>(h_k_in.data(), {batch_size, seq_len_k, hidden_size1}).swap_layout().shuffle(Eigen::array<int, 3>({2,1,0}));
    const Eigen::Tensor<float, 3> v_in = Eigen::TensorMap<const Eigen::Tensor<float, 3, Eigen::RowMajor>>(h_v_in.data(), {batch_size, seq_len_k, hidden_size1}).swap_layout().shuffle(Eigen::array<int, 3>({2,1,0}));

    const Eigen::Tensor<float, 3> target = Eigen::TensorMap<const Eigen::Tensor<float, 3, Eigen::RowMajor>>(h_target.data(), {batch_size, seq_len_q, hidden_size2}).swap_layout().shuffle(Eigen::array<int, 3>({2,1,0}));


    // no init
    Eigen::Tensor<float, 3> q_out(batch_size, seq_len_q, hidden_size2);
    Eigen::Tensor<float, 3> k_out(batch_size, seq_len_k, hidden_size2);
    Eigen::Tensor<float, 3> v_out(batch_size, seq_len_k, hidden_size2);
    Eigen::Tensor<float, 3> o_out(batch_size, seq_len_q, hidden_size2);
    Eigen::Tensor<float, 3> o_in(batch_size, seq_len_q, hidden_size2);

    
    Eigen::Tensor<float, 4> q(batch_size, n_heads, seq_len_q, head_size2); 
    Eigen::Tensor<float, 4> k(batch_size, n_heads, seq_len_k, head_size2);
    Eigen::Tensor<float, 4> v(batch_size, n_heads, seq_len_k, head_size2);
    
    Eigen::Tensor<float, 4> s(batch_size, n_heads, seq_len_q, seq_len_k);
    Eigen::Tensor<float, 4> p(batch_size, n_heads, seq_len_q, seq_len_k);
    Eigen::Tensor<float, 4> o(batch_size, n_heads, seq_len_q, head_size2);
    
    Eigen::Tensor<float, 3> q_out_grad(batch_size, seq_len_q, hidden_size2);
    Eigen::Tensor<float, 3> k_out_grad(batch_size, seq_len_k, hidden_size2);
    Eigen::Tensor<float, 3> v_out_grad(batch_size, seq_len_k, hidden_size2);
    Eigen::Tensor<float, 3> o_out_grad(batch_size, seq_len_q, hidden_size2);

    Eigen::Tensor<float, 3> q_in_grad(batch_size, seq_len_q, hidden_size1);
    Eigen::Tensor<float, 3> k_in_grad(batch_size, seq_len_k, hidden_size1);
    Eigen::Tensor<float, 3> v_in_grad(batch_size, seq_len_k, hidden_size1);
    Eigen::Tensor<float, 3> o_in_grad(batch_size, seq_len_q, hidden_size2);
    
    Eigen::Tensor<float, 2> q_weight_grad(hidden_size2, hidden_size1); 
    Eigen::Tensor<float, 2> k_weight_grad(hidden_size2, hidden_size1); 
    Eigen::Tensor<float, 2> v_weight_grad(hidden_size2, hidden_size1); 
    Eigen::Tensor<float, 2> o_weight_grad(hidden_size2, hidden_size2); 

    Eigen::Tensor<float, 1> q_bias_grad(hidden_size2); 
    Eigen::Tensor<float, 1> k_bias_grad(hidden_size2); 
    Eigen::Tensor<float, 1> v_bias_grad(hidden_size2); 
    Eigen::Tensor<float, 1> o_bias_grad(hidden_size2); 
    
    Eigen::Tensor<float, 4> q_grad(batch_size, n_heads, seq_len_q, head_size2);
    Eigen::Tensor<float, 4> k_grad(batch_size, n_heads, seq_len_k, head_size2);
    Eigen::Tensor<float, 4> v_grad(batch_size, n_heads, seq_len_k, head_size2);
    Eigen::Tensor<float, 4> s_grad(batch_size, n_heads, seq_len_q, seq_len_k); 
    Eigen::Tensor<float, 4> p_grad(batch_size, n_heads, seq_len_q, seq_len_k);
    Eigen::Tensor<float, 4> o_grad(batch_size, n_heads, seq_len_q, head_size2); 
    
    Eigen::Tensor<float, 0> loss;
    Eigen::Tensor<float, 3> d_loss(batch_size, seq_len_q, hidden_size2);

    // Linear Layer for Q, K and V, forward
    eigenDNN::eidnnLinearForward(handle, q_in, q_weight, q_bias, q_out);
    eigenDNN::eidnnLinearForward(handle, k_in, k_weight, k_bias, k_out);
    eigenDNN::eidnnLinearForward(handle, v_in, v_weight, v_bias, v_out);


    // reshape Q, K and V, [batch_size, seq_len, hidden_size] -> [batch_size, n_heads, seq_len, head_size]
    Eigen::Tensor<float, 3, Eigen::RowMajor> q_out_row = q_out.swap_layout().shuffle(Eigen::array<int, 3>({2,1,0}));
    Eigen::Tensor<float, 3, Eigen::RowMajor> k_out_row = k_out.swap_layout().shuffle(Eigen::array<int, 3>({2,1,0}));
    Eigen::Tensor<float, 3, Eigen::RowMajor> v_out_row = v_out.swap_layout().shuffle(Eigen::array<int, 3>({2,1,0}));
    Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>> q_0123_row(q_out_row.data(), {batch_size, seq_len_q, n_heads, head_size2});
    Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>> k_0123_row(k_out_row.data(), {batch_size, seq_len_k, n_heads, head_size2});
    Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>> v_0123_row(v_out_row.data(), {batch_size, seq_len_k, n_heads, head_size2});
    Eigen::Tensor<float, 4> q_0123 = q_0123_row.swap_layout().shuffle(Eigen::array<int, 4>({3,2,1,0}));
    Eigen::Tensor<float, 4> k_0123 = k_0123_row.swap_layout().shuffle(Eigen::array<int, 4>({3,2,1,0}));
    Eigen::Tensor<float, 4> v_0123 = v_0123_row.swap_layout().shuffle(Eigen::array<int, 4>({3,2,1,0}));
    q = q_0123.shuffle(Eigen::array<int, 4>({0,2,1,3}));
    k = k_0123.shuffle(Eigen::array<int, 4>({0,2,1,3}));
    v = v_0123.shuffle(Eigen::array<int, 4>({0,2,1,3}));

    // S = Q*K^T, forward
    eigenDNN::eidnnStridedBatchedGemmForward(handle, 1.0f/sqrtf(head_size2), 0, false, true, false, q, k, s); 

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
    Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>> o_in_row(o_0213_row.data(), {batch_size, seq_len_q, hidden_size2});
    o_in = o_in_row.swap_layout().shuffle(Eigen::array<int, 3>({2,1,0}));


    // Linear Layer for O, forward
    eigenDNN::eidnnLinearForward(handle, o_in, o_weight, o_bias, o_out);

    Eigen::Tensor<float, 3, Eigen::RowMajor> o_out_row = o_out.swap_layout().shuffle(Eigen::array<int, 3>({2,1,0}));
    // this->register_raw_test_data(o_out_row.data(), batch_size*seq_len_q*hidden_size, "output"); 

    if(is_train)
    {
      // MSE Loss
      eigenDNN::eidnnMSELoss(handle, o_out, target, loss, d_loss);

      // Linear Layer for O, backward
      o_out_grad = d_loss;
      eigenDNN::eidnnLinearBackward(handle, o_out_grad, o_in, o_weight, o_in_grad, o_weight_grad, o_bias_grad);

      // reshape O, [batch_size, seq_len, hidden_size] -> [batch_size, n_heads, seq_len, head_size]
      Eigen::Tensor<float, 3, Eigen::RowMajor> o_in_grad_row = o_in_grad.swap_layout().shuffle(Eigen::array<int, 3>({2,1,0}));
      Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>> o_in_grad_0123_row(o_in_grad_row.data(), {batch_size, seq_len_q, n_heads, head_size2});
      Eigen::Tensor<float, 4> o_in_grad_0123 = o_in_grad_0123_row.swap_layout().shuffle(Eigen::array<int, 4>({3,2,1,0}));
      o_grad = o_in_grad_0123.shuffle(Eigen::array<int, 4>({0,2,1,3}));

      // O=P*V backward
      eigenDNN::eidnnStridedBatchedGemmBackward(handle,  1, 0, false, false, false, p, v, o_grad, p_grad, v_grad);

      // P = dropout(P), backward
      eigenDNN::eidnnDropoutBackward(handle, dropoutDesc, p_grad, p_grad);

      // P = softmax(S), backward
      eigenDNN::eidnnSoftmaxBackward(handle, eigenDNN::eidnnSoftmaxAlgorithm_t::EIDNN_SOFTMAX_ACCURATE, eigenDNN::eidnnSoftmaxMode_t::EIDNN_SOFTMAX_MODE_INSTANCE, p, p_grad, s_grad);

      // S = Q*K^T, backward
      eigenDNN::eidnnStridedBatchedGemmBackward(handle,  1.0f/sqrtf(head_size2), 0, false, true, false, q, k, s_grad, q_grad, k_grad); 

      // reshape Q, K and V, [batch_size, n_heads, seq_len, head_size] -> [batch_size, seq_len, hidden_size] 
      Eigen::Tensor<float, 4> q_grad_0213 = q_grad.shuffle(Eigen::array<int, 4>({0,2,1,3}));
      Eigen::Tensor<float, 4> k_grad_0213 = k_grad.shuffle(Eigen::array<int, 4>({0,2,1,3}));
      Eigen::Tensor<float, 4> v_grad_0213 = v_grad.shuffle(Eigen::array<int, 4>({0,2,1,3}));
      Eigen::Tensor<float, 4, Eigen::RowMajor> q_grad_0213_row = q_grad_0213.swap_layout().shuffle(Eigen::array<int, 4>({3,2,1,0}));
      Eigen::Tensor<float, 4, Eigen::RowMajor> k_grad_0213_row = k_grad_0213.swap_layout().shuffle(Eigen::array<int, 4>({3,2,1,0}));
      Eigen::Tensor<float, 4, Eigen::RowMajor> v_grad_0213_row = v_grad_0213.swap_layout().shuffle(Eigen::array<int, 4>({3,2,1,0}));
      Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>> q_out_grad_row(q_grad_0213_row.data(), {batch_size, seq_len_q, hidden_size2});
      Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>> k_out_grad_row(k_grad_0213_row.data(), {batch_size, seq_len_k, hidden_size2});
      Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>> v_out_grad_row(v_grad_0213_row.data(), {batch_size, seq_len_k, hidden_size2});
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

      // this->register_raw_test_data(q_weight_grad_row.data(), hidden_size2*hidden_size1, "q_weight_grad");
      // this->register_raw_test_data(k_weight_grad_row.data(), hidden_size2*hidden_size1, "k_weight_grad");
      // this->register_raw_test_data(v_weight_grad_row.data(), hidden_size2*hidden_size1, "v_weight_grad");
      // this->register_raw_test_data(o_weight_grad_row.data(), hidden_size2*hidden_size2, "o_weight_grad");

      Eigen::Tensor<float, 1, Eigen::RowMajor> q_bias_grad_row = q_bias_grad.swap_layout().shuffle(Eigen::array<int, 1>({0}));
      Eigen::Tensor<float, 1, Eigen::RowMajor> k_bias_grad_row = k_bias_grad.swap_layout().shuffle(Eigen::array<int, 1>({0}));
      Eigen::Tensor<float, 1, Eigen::RowMajor> v_bias_grad_row = v_bias_grad.swap_layout().shuffle(Eigen::array<int, 1>({0}));
      Eigen::Tensor<float, 1, Eigen::RowMajor> o_bias_grad_row = o_bias_grad.swap_layout().shuffle(Eigen::array<int, 1>({0}));

      // this->register_raw_test_data(q_bias_grad_row.data(), hidden_size2, "q_bias_grad");
      // this->register_raw_test_data(k_bias_grad_row.data(), hidden_size2, "k_bias_grad");
      // this->register_raw_test_data(v_bias_grad_row.data(), hidden_size2, "v_bias_grad");
      // this->register_raw_test_data(o_bias_grad_row.data(), hidden_size2, "o_bias_grad");
    }
  }

  void run_cudnn_dnn(){
    // beamsize=1, 
    // for comparison, you have to set dropout=0
    testOpts opts;

    // Default test parameters to be overwritten by user cmd line options.
    opts.attnTrain       = is_train;
    opts.attnDataType    = CUDNN_DATA_FLOAT;
    opts.attnCompPrec    = CUDNN_DATA_FLOAT;
    opts.attnNumHeads    = n_heads;
    opts.attnBeamSize    = 1;
    opts.attnSmScaler    = 1.0;
    opts.attnDropoutRate = 0.0;
    opts.attnQsize       = hidden_size1;
    opts.attnKsize       = hidden_size1;
    opts.attnVsize       = hidden_size1;
    opts.attnProjQsize   = head_size2;
    opts.attnProjKsize   = head_size2;
    opts.attnProjVsize   = head_size2;
    opts.attnProjOsize   = head_size2;
    opts.attnSeqLenQ     = seq_len_q;
    opts.attnSeqLenK     = seq_len_k;
    opts.attnBatchSize   = batch_size;
    opts.attnResLink     = 0;
    opts.attnProjBias    = 0;

    if(opts.attnTrain == 0)
    {
      MultiheadAttentionTest<false, float,  float> attnTest;
      attnTest.setup(opts);
      attnTest.run();
      attnTest.teardown();
    }
    else if(opts.attnTrain == 1){
      MultiheadAttentionTest<true, float,  float> attnTest;
      attnTest.setup(opts);
      attnTest.run();
      attnTest.teardown();
    }
    printf("\nTest DONE\n\n");
    fflush(stdout);
  }



private:
  bool is_train;
  int batch_size, n_heads, seq_len_q, seq_len_k, head_size1, head_size2, hidden_size1, hidden_size2;
  float dropout_rate;

  std::vector<float> h_q_weight;
  std::vector<float> h_k_weight;
  std::vector<float> h_v_weight;
  std::vector<float> h_o_weight;

  std::vector<float> h_q_bias;
  std::vector<float> h_k_bias;
  std::vector<float> h_v_bias;
  std::vector<float> h_o_bias;

  std::vector<float> h_q_in;
  std::vector<float> h_k_in;
  std::vector<float> h_v_in;
  std::vector<float> h_target;
};

int eval_mha(unsigned batch_size,unsigned n_heads,unsigned seq_len_q,unsigned seq_len_k,unsigned head_size1,unsigned head_size2,float dropout_rate,bool is_train){
  test_MHA test_mha(batch_size,n_heads,seq_len_q,seq_len_k,head_size1,head_size2,dropout_rate,is_train);
  test_mha.init_data();
  test_mha.run_eigen_dnn();
  test_mha.run_cudnn_dnn();
}

TEST_CASE("MHA", "[mha]") {
  SECTION("[2,3,4,5,6,0.5]") {
    eval_mha(2,3,4,5,6,7,0,false);
  }
  SECTION("[2,3,4,5,6,0.5]") {
    eval_mha(2,3,4,5,6,7,0,true);
  }
  SECTION("[4,5,6,7,8,0.5]") {
    eval_mha(4,5,6,7,8,9,0,false);
  }
  SECTION("[4,5,6,7,8,0.5]") {
    eval_mha(4,5,6,7,8,9,0,true);
  }

}