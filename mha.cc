
#include "util.cuh"
  

  
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

  ~test_MHA(){
    cudaFree(devQ);
    devQ = NULL;

    cudaFree(devK);
    devK = NULL;

    cudaFree(devV);
    devV = NULL;

    cudaFree(devO);
    devO = NULL;

    cudaFree(devW);
    devW = NULL;

    cudaFree(devWkspace);
    devWkspace = NULL;

    cudaFree(devReserve);
    devReserve = NULL;

    free(hostO);
    hostO = NULL;

    if (is_train) {
        cudaFree(devTarget);
        devTarget = NULL;
        
        cudaFree(devDQ);
        devDQ = NULL;

        cudaFree(devDK);
        devDK = NULL;

        cudaFree(devDV);
        devDV = NULL;

        cudaFree(devDO);
        devDO = NULL;

        cudaFree(devDW);
        devDW = NULL;       
    
        free(hostDW);
        hostDW = NULL;

        free(hostDQ);
        hostDQ = NULL;

        free(hostDK);
        hostDK = NULL;

        free(hostDV);
        hostDV = NULL;
    }

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

    h_weight_bank = std::vector<float>(weight_len1*3+weight_len2);

    h_q_in = std::vector<float>(in_data_len_q);
    h_k_in = std::vector<float>(in_data_len_k);
    h_v_in = std::vector<float>(in_data_len_k);
    h_target = std::vector<float>(out_data_len);

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

    h_q_bias = std::vector<float>(bias_len,0);
    h_k_bias = std::vector<float>(bias_len,0);
    h_v_bias = std::vector<float>(bias_len,0);
    h_o_bias = std::vector<float>(bias_len,0);

    h_q_weight = std::vector<float>(h_weight_bank.begin(),h_weight_bank.begin()+weight_len1); 
    h_k_weight = std::vector<float>(h_weight_bank.begin()+weight_len1,h_weight_bank.begin()+weight_len1*2);
    h_v_weight = std::vector<float>(h_weight_bank.begin()+weight_len1*2,h_weight_bank.begin()+weight_len1*3);
    h_o_weight = std::vector<float>(h_weight_bank.begin()+weight_len1*3,h_weight_bank.begin()+weight_len1*3+weight_len2);

    printf("##### sizeWeights: %d\n",h_weight_bank.size()*sizeof(float));

  }

  
  void run_eigen_dnn(){
    using namespace Eigen;
    eigenDNN::eidnnHandle_t handle;
    void* saved_states;

    const Eigen::Tensor<float, 2> q_weight = Eigen::TensorMap<const Eigen::Tensor<float, 2>>(h_q_weight.data(), {hidden_size2, hidden_size1});
    const Eigen::Tensor<float, 2> k_weight = Eigen::TensorMap<const Eigen::Tensor<float, 2>>(h_k_weight.data(), {hidden_size2, hidden_size1});
    const Eigen::Tensor<float, 2> v_weight = Eigen::TensorMap<const Eigen::Tensor<float, 2>>(h_v_weight.data(), {hidden_size2, hidden_size1});
    const Eigen::Tensor<float, 2> o_weight = Eigen::TensorMap<const Eigen::Tensor<float, 2>>(h_o_weight.data(), {hidden_size2, hidden_size2});

    const Eigen::Tensor<float, 1> q_bias = Eigen::TensorMap<const Eigen::Tensor<float, 1>>(h_q_bias.data(), {hidden_size2});
    const Eigen::Tensor<float, 1> k_bias = Eigen::TensorMap<const Eigen::Tensor<float, 1>>(h_k_bias.data(), {hidden_size2});
    const Eigen::Tensor<float, 1> v_bias = Eigen::TensorMap<const Eigen::Tensor<float, 1>>(h_v_bias.data(), {hidden_size2});
    const Eigen::Tensor<float, 1> o_bias = Eigen::TensorMap<const Eigen::Tensor<float, 1>>(h_o_bias.data(), {hidden_size2});

    const Eigen::Tensor<float, 3> q_in = Eigen::TensorMap<const Eigen::Tensor<float, 3>>(h_q_in.data(), {batch_size, seq_len_q, hidden_size1});
    const Eigen::Tensor<float, 3> k_in = Eigen::TensorMap<const Eigen::Tensor<float, 3>>(h_k_in.data(), {batch_size, seq_len_k, hidden_size1});
    const Eigen::Tensor<float, 3> v_in = Eigen::TensorMap<const Eigen::Tensor<float, 3>>(h_v_in.data(), {batch_size, seq_len_k, hidden_size1});

    const Eigen::Tensor<float, 3> target = Eigen::TensorMap<const Eigen::Tensor<float, 3>>(h_target.data(), {batch_size, seq_len_q, hidden_size2});


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

    // std::cout << "s: " << s << std::endl;

    // P = softmax(S), forward
    eigenDNN::eidnnSoftmaxForward(handle, eigenDNN::eidnnSoftmaxAlgorithm_t::EIDNN_SOFTMAX_ACCURATE, eigenDNN::eidnnSoftmaxMode_t::EIDNN_SOFTMAX_MODE_INSTANCE, s, p);

    // P = dropout(P), forward
    eigenDNN::eidnnDropoutDescriptor_t dropoutDesc = std::make_tuple(dropout_rate,saved_states,0,2023);
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
    h_o_out.assign(o_out_row.data(),o_out_row.data()+batch_size*seq_len_q*hidden_size2);

    // h_o_out.assign(o_in_row.data(),o_in_row.data()+batch_size*seq_len_q*hidden_size2);

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

        h_q_in_grad.assign(q_in_grad.data(),q_in_grad.data()+batch_size*seq_len_q*hidden_size1);
        h_k_in_grad.assign(k_in_grad.data(),k_in_grad.data()+batch_size*seq_len_k*hidden_size1);
        h_v_in_grad.assign(v_in_grad.data(),v_in_grad.data()+batch_size*seq_len_k*hidden_size1);

        h_q_weight_grad.assign(q_weight_grad.data(),q_weight_grad.data()+hidden_size2*hidden_size1);
        h_k_weight_grad.assign(k_weight_grad.data(),k_weight_grad.data()+hidden_size2*hidden_size1);
        h_v_weight_grad.assign(v_weight_grad.data(),v_weight_grad.data()+hidden_size2*hidden_size1);
        h_o_weight_grad.assign(o_weight_grad.data(),o_weight_grad.data()+hidden_size2*hidden_size2);
    }
  }


  void run_cudnn_dnn(){
    // beamsize=1, bias=false
    // for comparison, you have to set dropout=0
    // Default test parameters to be overwritten by user cmd line options.
    int numHeads    = n_heads;
    int beamSize    = 1;
    double smScaler    = 1.0f/sqrtf(head_size2);
    float dropoutRate = dropout_rate;
    int qSize       = hidden_size1;
    int kSize       = hidden_size1;
    int vSize       = hidden_size1;
    int qProjSize   = head_size2;
    int kProjSize   = head_size2;
    int vProjSize   = head_size2;
    int oProjSize   = hidden_size2;
    int seqLenQ     = seq_len_q;
    int seqLenK     = seq_len_k;
    int batchSize   = batch_size;
    bool resLink     = false;
    bool projBias    = false;

    std::cout << "qSize" << qSize << std::endl;
    std::cout << "kSize" << kSize << std::endl;
    std::cout << "vSize" << vSize << std::endl;
    std::cout << "qProjSize" << qProjSize << std::endl;
    std::cout << "kProjSize" << kProjSize << std::endl;
    std::cout << "vProjSize" << vProjSize << std::endl;
    std::cout << "oProjSize" << oProjSize << std::endl;


    cudnnHandle_t handle;
    cudnnAttnDescriptor_t attn_desc;
    cudnnDropoutDescriptor_t drop_desc;
    cudnnSeqDataDescriptor_t q_desc;
    cudnnSeqDataDescriptor_t k_desc;
    cudnnSeqDataDescriptor_t v_desc;
    cudnnSeqDataDescriptor_t o_desc;
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT; 
    cudnnDataType_t compPrec = CUDNN_DATA_FLOAT;

    CHECK_CUDNN_ERR(cudnnCreate(&handle));
    CHECK_CUDNN_ERR(cudnnCreateAttnDescriptor(&attn_desc));
    CHECK_CUDNN_ERR(cudnnCreateDropoutDescriptor(&drop_desc));
    CHECK_CUDNN_ERR(cudnnCreateSeqDataDescriptor(&q_desc));
    CHECK_CUDNN_ERR(cudnnCreateSeqDataDescriptor(&k_desc));
    CHECK_CUDNN_ERR(cudnnCreateSeqDataDescriptor(&v_desc));
    CHECK_CUDNN_ERR(cudnnCreateSeqDataDescriptor(&o_desc));

    size_t dropoutBufSize;
    void *dropoutBuf;


    int* qSeqArray = NULL;
    int* kSeqArray = NULL;

    int *loWinIdx = NULL;
    int *hiWinIdx = NULL;

    unsigned attnMode = 0;
    if (projBias == 0) {
        attnMode = (attnMode | CUDNN_ATTN_DISABLE_PROJ_BIASES | CUDNN_ATTN_QUERYMAP_ALL_TO_ONE);
    } else if (projBias == 1) {
        attnMode = (attnMode | CUDNN_ATTN_ENABLE_PROJ_BIASES | CUDNN_ATTN_QUERYMAP_ALL_TO_ONE);
    } else {
        fprintf(stderr, "ERROR: wrong -attnProjBias value\n\n");
        exit(-1);
    }

    if (numHeads <= 0 || batchSize <= 0 || beamSize <= 0) {
        fprintf(stderr, "ERROR: wrong attention NumHeads/BatchSize/BeamSize arguments\n\n");
        exit(-1);
    }

    int oSize = vProjSize > 0 ? vProjSize * numHeads : vSize;

    size_t qoTokens = size_t(seqLenQ) * batchSize * beamSize;
    size_t kvTokens = size_t(seqLenK) * batchSize;
    
    size_t qNmbElem = qoTokens * qSize;
    size_t kNmbElem = kvTokens * kSize;
    size_t vNmbElem = kvTokens * vSize;
    size_t oNmbElem = qoTokens * oSize;

    size_t qNmbWeights = (qProjSize > 0 ? size_t(qSize) * qProjSize : 0) * numHeads;
    size_t kNmbWeights = (kProjSize > 0 ? size_t(kSize) * kProjSize : 0) * numHeads;
    size_t vNmbWeights = (vProjSize > 0 ? size_t(vSize) * vProjSize : 0) * numHeads;
    size_t oNmbWeights = (oProjSize > 0 ? size_t(oSize) * oProjSize : 0);

    if (qNmbElem == 0 || kNmbElem == 0 || oNmbElem == 0) {
        fprintf(stderr, "ERROR: Q/K/O data buffers cannot be zero size\n\n");
        exit(-1);
    }

    // Allocate input and output buffers (forward/inference pass).
    CHECK_CUDA_ERR(cudaMalloc((void **)&devQ, qNmbElem * sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc((void **)&devK, kNmbElem * sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc((void **)&devV, vNmbElem * sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc((void **)&devO, oNmbElem * sizeof(float)));

    // Allocate input and output buffers (backward/training pass).
    if (is_train) {
        CHECK_CUDA_ERR(cudaMalloc((void **)&devTarget, oNmbElem * sizeof(float)));
        CHECK_CUDA_ERR(cudaMalloc((void **)&devDQ, qNmbElem * sizeof(float)));
        CHECK_CUDA_ERR(cudaMalloc((void **)&devDK, kNmbElem * sizeof(float)));
        CHECK_CUDA_ERR(cudaMalloc((void **)&devDV, vNmbElem * sizeof(float)));
        CHECK_CUDA_ERR(cudaMalloc((void **)&devDO, oNmbElem * sizeof(float)));
    }

    CHECK_CUDNN_ERR(cudnnDropoutGetStatesSize(handle, &dropoutBufSize));
    CHECK_CUDA_ERR(cudaMalloc((void **)&dropoutBuf, dropoutBufSize));

    CHECK_CUDNN_ERR(cudnnSetDropoutDescriptor(drop_desc, handle, dropoutRate, dropoutBuf, dropoutBufSize, 0));



    CHECK_CUDNN_ERR(cudnnSetAttnDescriptor(attn_desc, // 
                                           attnMode,
                                           numHeads,
                                           smScaler,
                                           dataType,
                                           compPrec,
                                           CUDNN_DEFAULT_MATH,
                                           is_train && dropoutRate > 0.0 ? drop_desc : NULL,
                                           NULL,
                                           qSize,
                                           kSize,
                                           vSize,
                                           qProjSize,
                                           kProjSize,
                                           vProjSize,
                                           oProjSize,
                                           seqLenQ,
                                           seqLenK,
                                           batchSize,
                                           beamSize));

    size_t sizeWeights = 0, sizeWkspace = 0, sizeReserve = 0;
    if (is_train) {
        CHECK_CUDNN_ERR(cudnnGetMultiHeadAttnBuffers(handle, attn_desc, &sizeWeights, &sizeWkspace, &sizeReserve));
    } else {
        CHECK_CUDNN_ERR(cudnnGetMultiHeadAttnBuffers(handle, attn_desc, &sizeWeights, &sizeWkspace, NULL));
    }

    printf("@@@@@ sizeWeights: %d\n",sizeWeights);
    printf("@@@@@ sizeWkspace: %d\n",sizeWkspace);
    printf("@@@@@ sizeReserve: %d\n",sizeReserve);
    

    if (sizeWeights > 0) {
        CHECK_CUDA_ERR(cudaMalloc((void **)&devW, sizeWeights));
        if (is_train) {
            CHECK_CUDA_ERR(cudaMalloc((void **)&devDW, sizeWeights));
        }
    }
    if (sizeWkspace > 0) {
        CHECK_CUDA_ERR(cudaMalloc((void **)&devWkspace, sizeWkspace));
    }
    if (sizeReserve > 0) {
        CHECK_CUDA_ERR(cudaMalloc((void **)&devReserve, sizeReserve));

        // Fill with -NaN to deterct incorrect segment write for debugging.
        CHECK_CUDA_ERR(cudaMemset(devReserve, 0xff, sizeReserve));
    }


    // get the Weight Info
    cudnnTensorDescriptor_t weightDesc = NULL;
    int nbDims, dimW[4], strideW[4];
    cudnnDataType_t dataTypeUnsed;

    CHECK_CUDNN_ERR(cudnnCreateTensorDescriptor(&weightDesc));
        
    float *weightAddr = NULL;
    void *paramBuf;
    cudnnMultiHeadAttnWeightKind_t wKind[4] = {CUDNN_MH_ATTN_Q_WEIGHTS, CUDNN_MH_ATTN_K_WEIGHTS, CUDNN_MH_ATTN_V_WEIGHTS, CUDNN_MH_ATTN_O_WEIGHTS};
    std::vector<std::string> wKindNames({"CUDNN_MH_ATTN_Q_WEIGHTS", "CUDNN_MH_ATTN_K_WEIGHTS", "CUDNN_MH_ATTN_V_WEIGHTS", "CUDNN_MH_ATTN_O_WEIGHTS"});
    for(int i=0; i<4; i++){
        size_t paramSize = sizeWeights;
        CHECK_CUDNN_ERR(cudnnGetMultiHeadAttnWeights(handle, attn_desc, wKind[i], paramSize, paramBuf, weightDesc, (void **)&weightAddr));
        CHECK_CUDNN_ERR(cudnnGetTensorNdDescriptor(weightDesc, 4, &dataTypeUnsed, &nbDims, dimW, strideW));
        printf("@@@@@ [%s] weightAddr %p\n", wKindNames[i].c_str(), (void *)weightAddr);
        printf("@@@@@ [%s] dimW[0] %d  dimW[1] %d  dimW[2] %d  dimW[3] %d\n", wKindNames[i].c_str(), dimW[0], dimW[1], dimW[2], dimW[3]);
        printf("@@@@@ [%s] strideW[0] %d  strideW[1] %d  strideW[2] %d  strideW[3] %d\n", wKindNames[i].c_str(), strideW[0], strideW[1], strideW[2], strideW[3]);
    }

    qSeqArray = (int *)calloc(batchSize * beamSize, sizeof(int));
    kSeqArray = (int *)calloc(batchSize, sizeof(int));


    if (loWinIdx == NULL && hiWinIdx == NULL) {
        loWinIdx = (int *)calloc(seqLenQ, sizeof(int));
        hiWinIdx = (int *)calloc(seqLenQ, sizeof(int));
    }

    // Allocate weight and data buffers on the CPU side.
    if (sizeWeights > 0) {
        hostDW = (float *)malloc(sizeWeights);
    }

    hostO = (float *)malloc(oNmbElem * sizeof(float));

    // Allocate input and output buffers (backward/training pass).
    if (is_train) {
        hostDQ = (float *)malloc(qNmbElem * sizeof(float));
        hostDK = (float *)malloc(kNmbElem * sizeof(float));
        hostDV = (float *)malloc(vNmbElem * sizeof(float));
    }



    /*********************/

    // Initialize qSeqArray and kSeqArray values and attention window
    size_t qBatches = batchSize * beamSize;
    size_t kBatches = batchSize * 1;

    // Fixed lengths for all sequences in a batch.
    for (size_t i = 0; i < qBatches; ++i) {
        qSeqArray[i] = seqLenQ;
    }

    for (size_t i = 0; i < kBatches; ++i) {
        kSeqArray[i] = seqLenK;
    }

    // Set the maximum attention window in all time-steps.
    for (int i = 0; i < seqLenQ; ++i) {
        loWinIdx[i] = 0;
        hiWinIdx[i] = seqLenK;
    }
    

    printf("Test parameters:\n\n");
    printf("#### attnTrain       = %d (%s)\n", is_train, is_train ? "training" : "inference");
    printf("#### attnDataType    = %d (FP%d)\n", dataType, int(8*sizeof(float)));
    printf("#### attnCompPrec    = %d (FP%d)\n", compPrec, int(8*sizeof(float)));
    printf("#### attnNumHeads    = %d\n", numHeads);
    printf("#### attnBatchSize   = %d\n", batchSize);
    printf("#### attnBeamSize    = %d\n", beamSize);
    printf("#### attnSmScaler    = %.4e\n", smScaler);
    printf("#### attnDropoutRate = %.4f\n", dropoutRate);
    printf("#### attnQsize       = %d\n", qSize);
    printf("#### attnKsize       = %d\n", kSize);
    printf("#### attnVsize       = %d\n", vSize);
    printf("#### attnProjQsize   = %d%s\n", qProjSize, qProjSize ? "" : " (no Q weights)");
    printf("#### attnProjKsize   = %d%s\n", kProjSize, kProjSize ? "" : " (no K weights)");
    printf("#### attnProjVsize   = %d%s\n", vProjSize, vProjSize ? "" : " (no V weights)");
    printf("#### attnProjOsize   = %d%s\n", oProjSize, oProjSize ? "" : " (no O weights)");
    printf("#### attnSeqLenQ     = %d\n", seqLenQ);
    printf("#### attnSeqLenK     = %d\n", seqLenK);
    printf("#### attnResLink     = %d\n", resLink);
    printf("#### attnProjBias    = %d\n", projBias);

    for (size_t i = 0; i < qBatches; ++i) {
        printf("sequence_length_q[idx=%lu]=%d\n", i, qSeqArray[i]);
    }
    printf("\n");

    for (size_t i = 0; i < kBatches; ++i) {
        printf("sequence_length_k[idx=%lu]=%d\n", i, kSeqArray[i]);
    }
    printf("\n");

    for (int i = 0; i < seqLenQ; ++i) {
        printf("attention_window[time=%d]=%d:%d\n", i, loWinIdx[i], hiWinIdx[i]);
    }
    printf("\n");

    /*********************/

    int qSeqArraySize = beamSize * batchSize;
    int kSeqArraySize = batchSize;

    // host-to-device copies
    size_t size = sizeof(int) * qSeqArraySize;
    int *devQSeqArray;
    int *devKSeqArray;
    CHECK_CUDA_ERR(cudaMalloc((void **)&devQSeqArray, size));
    CHECK_CUDA_ERR(cudaMemcpy(devQSeqArray, qSeqArray, size, cudaMemcpyHostToDevice));

    size = sizeof(int) * kSeqArraySize;
    CHECK_CUDA_ERR(cudaMalloc((void **)&devKSeqArray, size));
    CHECK_CUDA_ERR(cudaMemcpy(devKSeqArray, kSeqArray, size, cudaMemcpyHostToDevice));


    int dimA[CUDNN_SEQDATA_DIM_COUNT];
    
    cudnnSeqDataAxis_t dataAxes[CUDNN_SEQDATA_DIM_COUNT];
    dataAxes[0] = CUDNN_SEQDATA_BEAM_DIM;
    dataAxes[1] = CUDNN_SEQDATA_BATCH_DIM;
    dataAxes[2] = CUDNN_SEQDATA_TIME_DIM;
    dataAxes[3] = CUDNN_SEQDATA_VECT_DIM;

    dimA[CUDNN_SEQDATA_BEAM_DIM]  = beamSize;
    dimA[CUDNN_SEQDATA_BATCH_DIM] = batchSize;
    dimA[CUDNN_SEQDATA_TIME_DIM]  = seqLenQ;
    dimA[CUDNN_SEQDATA_VECT_DIM]  = qSize;
    CHECK_CUDNN_ERR(cudnnSetSeqDataDescriptor(q_desc, dataType, CUDNN_SEQDATA_DIM_COUNT, dimA, dataAxes, qSeqArraySize, qSeqArray, NULL));

    dimA[CUDNN_SEQDATA_BEAM_DIM]  = beamSize;
    dimA[CUDNN_SEQDATA_BATCH_DIM] = batchSize;
    dimA[CUDNN_SEQDATA_TIME_DIM]  = seqLenQ;
    dimA[CUDNN_SEQDATA_VECT_DIM]  = oSize;
    CHECK_CUDNN_ERR(cudnnSetSeqDataDescriptor(o_desc, dataType, CUDNN_SEQDATA_DIM_COUNT, dimA, dataAxes, qSeqArraySize, qSeqArray, NULL));

    // seq-k
    dimA[CUDNN_SEQDATA_BEAM_DIM]  = 1;
    dimA[CUDNN_SEQDATA_BATCH_DIM] = batchSize;
    dimA[CUDNN_SEQDATA_TIME_DIM]  = seqLenK;
    dimA[CUDNN_SEQDATA_VECT_DIM]  = kSize;
    CHECK_CUDNN_ERR(cudnnSetSeqDataDescriptor(k_desc, dataType, CUDNN_SEQDATA_DIM_COUNT, dimA, dataAxes, kSeqArraySize, kSeqArray, NULL));

    // seq-v
    dimA[CUDNN_SEQDATA_BEAM_DIM]  = 1;
    dimA[CUDNN_SEQDATA_BATCH_DIM] = batchSize;
    dimA[CUDNN_SEQDATA_TIME_DIM]  = seqLenK;
    dimA[CUDNN_SEQDATA_VECT_DIM]  = vSize;
    CHECK_CUDNN_ERR(cudnnSetSeqDataDescriptor(v_desc, dataType, CUDNN_SEQDATA_DIM_COUNT, dimA, dataAxes, kSeqArraySize, kSeqArray, NULL));

    // Fill output surface with NaN-s.
    CHECK_CUDA_ERR(cudaMemset(devO, 0xFF, oNmbElem * sizeof(float)));

    if (is_train) {
        // Fill output surfaces with NaN-s.
        CHECK_CUDA_ERR(cudaMemset(devTarget, 0, sizeof(float) * oNmbElem));
        CHECK_CUDA_ERR(cudaMemset(devDQ, 0xFF, sizeof(float) * qNmbElem));
        CHECK_CUDA_ERR(cudaMemset(devDK, 0xFF, sizeof(float) * kNmbElem));
        CHECK_CUDA_ERR(cudaMemset(devDV, 0xFF, sizeof(float) * vNmbElem));
        CHECK_CUDA_ERR(cudaMemset(devDO, 0xFF, sizeof(float) * oNmbElem));

        // Fill the "wgrad" buffer with zeros (results are added to existing values).
        CHECK_CUDA_ERR(cudaMemset(devDW, 0, sizeWeights));
    }

    /* @junda: do not use the following lines for transposition */
    // std::vector<float> h_q_weight = vector01(this->h_q_weight,hidden_size2,hidden_size1);
    // std::vector<float> h_k_weight = vector01(this->h_k_weight,hidden_size2,hidden_size1);
    // std::vector<float> h_v_weight = vector01(this->h_v_weight,hidden_size2,hidden_size1);
    // std::vector<float> h_o_weight = vector01(this->h_o_weight,hidden_size2,hidden_size2);

    std::vector<float> h_q_in = vector0132(this->h_q_in,1,batch_size,hidden_size1,seq_len_q);
    std::vector<float> h_k_in = vector0132(this->h_k_in,1,batch_size,hidden_size1,seq_len_k);
    std::vector<float> h_v_in = vector0132(this->h_v_in,1,batch_size,hidden_size1,seq_len_k);
    

    // Copy the data from GPU (device) to CPU (host)
    CHECK_CUDA_ERR(cudaMemcpy(devW, h_weight_bank.data(), sizeWeights, cudaMemcpyHostToDevice)); /* @junda: the following lines are equivalent to this line */
    // CHECK_CUDA_ERR(cudaMemcpy(devW, h_q_weight.data(), h_q_weight.size()*sizeof(float), cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(devW + h_q_weight.size(), h_k_weight.data(), h_k_weight.size()*sizeof(float), cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(devW + h_q_weight.size() + h_k_weight.size(), h_v_weight.data(), h_v_weight.size()*sizeof(float), cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(devW + h_q_weight.size() + h_k_weight.size() + h_v_weight.size(), h_o_weight.data(), h_o_weight.size()*sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(devQ, h_q_in.data(), sizeof(float) * qNmbElem, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(devK, h_k_in.data(), sizeof(float) * kNmbElem, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(devV, h_v_in.data(), sizeof(float) * vNmbElem, cudaMemcpyHostToDevice));


    cudnnSeqDataAxis_t ordA[CUDNN_SEQDATA_DIM_COUNT];
    
    CHECK_CUDNN_ERR(cudnnGetSeqDataDescriptor(q_desc, NULL, NULL, 4, dimW, ordA, NULL, 0, NULL, NULL));
    printf("@@@@@ q_desc dimW[0] %d  dimW[1] %d  dimW[2] %d  dimW[3] %d\n", dimW[0], dimW[1], dimW[2], dimW[3]);
    printf("@@@@@ q_desc ordA[0] %d  ordA[1] %d  ordA[2] %d  ordA[3] %d\n", ordA[0], ordA[1], ordA[2], ordA[3]);

    CHECK_CUDNN_ERR(cudnnGetSeqDataDescriptor(k_desc, NULL, NULL, 4, dimW, ordA, NULL, 0, NULL, NULL));
    printf("@@@@@ k_desc dimW[0] %d  dimW[1] %d  dimW[2] %d  dimW[3] %d\n", dimW[0], dimW[1], dimW[2], dimW[3]);
    printf("@@@@@ k_desc ordA[0] %d  ordA[1] %d  ordA[2] %d  ordA[3] %d\n", ordA[0], ordA[1], ordA[2], ordA[3]);

    CHECK_CUDNN_ERR(cudnnGetSeqDataDescriptor(v_desc, NULL, NULL, 4, dimW, ordA, NULL, 0, NULL, NULL));
    printf("@@@@@ v_desc dimW[0] %d  dimW[1] %d  dimW[2] %d  dimW[3] %d\n", dimW[0], dimW[1], dimW[2], dimW[3]);
    printf("@@@@@ v_desc ordA[0] %d  ordA[1] %d  ordA[2] %d  ordA[3] %d\n", ordA[0], ordA[1], ordA[2], ordA[3]);

    CHECK_CUDNN_ERR(cudnnGetSeqDataDescriptor(o_desc, NULL, NULL, 4, dimW, ordA, NULL, 0, NULL, NULL));
    printf("@@@@@ o_desc dimW[0] %d  dimW[1] %d  dimW[2] %d  dimW[3] %d\n", dimW[0], dimW[1], dimW[2], dimW[3]);
    printf("@@@@@ o_desc ordA[0] %d  ordA[1] %d  ordA[2] %d  ordA[3] %d\n", ordA[0], ordA[1], ordA[2], ordA[3]);

    if (is_train) {
        if (sizeReserve == 0) {
            fprintf(stderr, "ERROR: zero reserve buffer size in training mode\n\n");
            exit(-1);
        }

        std::vector<float> h_target = vector0132(this->h_target,1,batch_size,hidden_size2,seq_len_q);
        CHECK_CUDA_ERR(cudaMemcpy(devTarget, h_target.data(), oNmbElem * sizeof(float), cudaMemcpyHostToDevice));

        printf("Calling cudnnMultiHeadAttnForward(currIdx = -1)\n");
        CHECK_CUDNN_ERR(cudnnMultiHeadAttnForward(handle,
                                                  attn_desc,
                                                  -1,
                                                  loWinIdx,
                                                  hiWinIdx,
                                                  devQSeqArray,
                                                  devKSeqArray,
                                                  q_desc,
                                                  devQ,
                                                  resLink ? devQ : NULL,
                                                  k_desc,
                                                  devK,
                                                  v_desc,
                                                  devV,
                                                  o_desc,
                                                  devO,
                                                  sizeWeights,
                                                  sizeWeights > 0 ? devW : NULL,
                                                  sizeWkspace,
                                                  devWkspace,
                                                  sizeReserve,
                                                  devReserve));
        
        float* d_loss;
        CHECK_CUDA_ERR(cudaMalloc((void**)&d_loss, sizeof(float)));
        launch_mse_loss_kernel(devO,devTarget,d_loss,devDO,oNmbElem);

        printf("Calling cudnnMultiHeadAttnBackwardData()\n");
        CHECK_CUDNN_ERR(cudnnMultiHeadAttnBackwardData(handle,
                                                       attn_desc,
                                                       loWinIdx,
                                                       hiWinIdx,
                                                       devQSeqArray,
                                                       devKSeqArray,
                                                       o_desc,
                                                       devDO,
                                                       q_desc,
                                                       devDQ,
                                                       devQ,
                                                       k_desc,
                                                       devDK,
                                                       devK,
                                                       v_desc,
                                                       devDV,
                                                       devV,
                                                       sizeWeights,
                                                       sizeWeights > 0 ? devW : NULL,
                                                       sizeWkspace,
                                                       devWkspace,
                                                       sizeReserve,
                                                       devReserve));
        
        printf("Calling cudnnMultiHeadAttnBackwardWeights()\n");
        CHECK_CUDNN_ERR(cudnnMultiHeadAttnBackwardWeights(handle,
                                                          attn_desc,
                                                          CUDNN_WGRAD_MODE_ADD,
                                                          q_desc,
                                                          devQ,
                                                          k_desc,
                                                          devK,
                                                          v_desc,
                                                          devV,
                                                          o_desc,
                                                          devDO,
                                                          sizeWeights,
                                                          sizeWeights > 0 ? devW : NULL,
                                                          sizeWeights > 0 ? devDW : NULL,
                                                          sizeWkspace,
                                                          devWkspace,
                                                          sizeReserve,
                                                          devReserve));
    } else {
        if (sizeReserve != 0) {
            fprintf(stderr, "ERROR: non-zero reserve buffer size in inference mode\n\n");
            exit(-1);
        }

        printf("Calling cudnnMultiHeadAttnForward(currIdx = -1)\n");
        CHECK_CUDNN_ERR(cudnnMultiHeadAttnForward(handle,
                                                    attn_desc,
                                                    -1,
                                                    loWinIdx,
                                                    hiWinIdx,
                                                    devQSeqArray,
                                                    devKSeqArray,
                                                    q_desc,
                                                    devQ,
                                                    resLink ? devQ : NULL,
                                                    k_desc,
                                                    devK,
                                                    v_desc,
                                                    devV,
                                                    o_desc,
                                                    devO,
                                                    sizeWeights,
                                                    sizeWeights > 0 ? devW : NULL,
                                                    sizeWkspace,
                                                    devWkspace,
                                                    0,
                                                    NULL));
    }

    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    // Copy forward output to host.
    CHECK_CUDA_ERR(cudaMemcpy(hostO, devO, oNmbElem * sizeof(float), cudaMemcpyDeviceToHost));

    if (is_train) {
        CHECK_CUDA_ERR(cudaMemcpy(hostDQ, devDQ, sizeof(float) * qNmbElem, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(hostDK, devDK, sizeof(float) * kNmbElem, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(hostDV, devDV, sizeof(float) * vNmbElem, cudaMemcpyDeviceToHost));

        std::vector<float> vec_hostDQ(hostDQ,hostDQ+qNmbElem);
        std::vector<float> vec_hostDQ_trans = vector0132(vec_hostDQ,1,batch_size,seq_len_q,hidden_size1);
        CHECK_CUDA_ERR(cudaMemcpy(hostDQ, vec_hostDQ_trans.data(), sizeof(float) * qNmbElem, cudaMemcpyHostToHost));

        std::vector<float> vec_hostDK(hostDK,hostDK+kNmbElem);
        std::vector<float> vec_hostDK_trans = vector0132(vec_hostDK,1,batch_size,seq_len_k,hidden_size1);
        CHECK_CUDA_ERR(cudaMemcpy(hostDK, vec_hostDK_trans.data(), sizeof(float) * kNmbElem, cudaMemcpyHostToHost));

        std::vector<float> vec_hostDV(hostDV,hostDV+vNmbElem);
        std::vector<float> vec_hostDV_trans = vector0132(vec_hostDV,1,batch_size,seq_len_k,hidden_size1);
        CHECK_CUDA_ERR(cudaMemcpy(hostDV, vec_hostDV_trans.data(), sizeof(float) * vNmbElem, cudaMemcpyHostToHost));

        // Copy wgrad results to host
        if (sizeWeights > 0) {
            CHECK_CUDA_ERR(cudaMemcpy(hostDW, devDW, sizeWeights, cudaMemcpyDeviceToHost));
        }
    }


    /******************************/

    cudnnDestroyAttnDescriptor(attn_desc);
    attn_desc = NULL;

    cudnnDestroyDropoutDescriptor(drop_desc);
    drop_desc = NULL;

    cudnnDestroySeqDataDescriptor(q_desc);
    q_desc = NULL;

    cudnnDestroySeqDataDescriptor(k_desc);
    k_desc = NULL;

    cudnnDestroySeqDataDescriptor(v_desc);
    v_desc = NULL;

    cudnnDestroySeqDataDescriptor(o_desc);
    o_desc = NULL;

    cudaFree(dropoutBuf);
    dropoutBuf = NULL;

    
    free(qSeqArray);
    qSeqArray = NULL;

    free(kSeqArray);
    kSeqArray = NULL;

    free(loWinIdx);
    loWinIdx = NULL;

    free(hiWinIdx);
    hiWinIdx = NULL;

    printf("\nTest DONE\n\n");
    fflush(stdout);
  }

  void verify(){
    size_t weight_len1 = hidden_size1*hidden_size2;
    size_t weight_len2 = hidden_size2*hidden_size2;
    size_t in_data_len_q = batch_size*seq_len_q*hidden_size1;
    size_t in_data_len_k = batch_size*seq_len_k*hidden_size1;
    size_t out_data_len = batch_size*seq_len_q*hidden_size2;

    
    if(!compareResults(hostO,h_o_out.data(),out_data_len))
    {
        print_vec(hostO,"cudnn O",0,h_o_out.size());
        print_vec(h_o_out.data(),"eidnn O",0,h_o_out.size());
    }

    if (is_train) {
        if(!compareResults(hostDQ,h_q_in_grad.data(),in_data_len_q))
        {
            print_vec(hostDQ,"cudnn dQ",0,64);
            print_vec(h_q_in_grad.data(),"eidnn dQ",0,64);
        }
        if(!compareResults(hostDK,h_k_in_grad.data(),in_data_len_k))
        {
            print_vec(hostDK,"cudnn dK",0,64);
            print_vec(h_k_in_grad.data(),"eidnn dK",0,64);
        }
        if(!compareResults(hostDV,h_v_in_grad.data(),in_data_len_k))
        {
            print_vec(hostDV,"cudnn dV",0,64);
            print_vec(h_v_in_grad.data(),"eidnn dV",0,64);
        }

        if(!compareResults(hostDW+0,h_q_weight_grad.data(),weight_len1))
        {
            print_vec(hostDW+0,"cudnn dQW",0,64);
            print_vec(h_q_weight_grad.data(),"eidnn dQW",0,64);
        }

        if(!compareResults(hostDW+weight_len1,h_k_weight_grad.data(),weight_len1))
        {
            print_vec(hostDW+weight_len1,"cudnn dKW",0,64);
            print_vec(h_k_weight_grad.data(),"eidnn dKW",0,64);
        }
        if(!compareResults(hostDW+2*weight_len1,h_v_weight_grad.data(),weight_len1))
        {
            print_vec(hostDW+2*weight_len1,"cudnn dVW",0,64);
            print_vec(h_v_weight_grad.data(),"eidnn dVW",0,64);
        }
        if(!compareResults(hostDW+3*weight_len1,h_o_weight_grad.data(),weight_len2))
        {
            print_vec(hostDW+3*weight_len1,"cudnn dOW",0,64);
            print_vec(h_o_weight_grad.data(),"eidnn dOW",0,64);
        }
    }

  }

private:
  bool is_train;
  int batch_size, n_heads, seq_len_q, seq_len_k, head_size1, head_size2, hidden_size1, hidden_size2;
  float dropout_rate;

  std::vector<float> h_weight_bank;
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



  float* devQ  = NULL;
  float* devK  = NULL;
  float* devV  = NULL;
  float* devO  = NULL;
  float* devTarget = NULL;

  float* devDQ = NULL;
  float* devDK = NULL;
  float* devDV = NULL;
  float* devDO = NULL;
  float* devW  = NULL;
  float* devDW = NULL;
  float* devWkspace = NULL;
  float* devReserve = NULL;

  
    std::vector<float> h_o_out;
    std::vector<float> h_q_in_grad;
    std::vector<float> h_k_in_grad;
    std::vector<float> h_v_in_grad;

    std::vector<float> h_q_weight_grad;
    std::vector<float> h_k_weight_grad;
    std::vector<float> h_v_weight_grad;
    std::vector<float> h_o_weight_grad;

    float* hostO  = NULL;
    float* hostDQ = NULL;
    float* hostDK = NULL;
    float* hostDV = NULL;
    float* hostDW = NULL;
};

int eval_mha(unsigned batch_size,unsigned n_heads,unsigned seq_len_q,unsigned seq_len_k,unsigned head_size1,unsigned head_size2,float dropout_rate,bool is_train){
  test_MHA test_mha(batch_size,n_heads,seq_len_q,seq_len_k,head_size1,head_size2,dropout_rate,is_train);
  test_mha.init_data();
  test_mha.run_eigen_dnn();   
  test_mha.run_cudnn_dnn();
  test_mha.verify();
}

int main(){
    eval_mha(1,4,2,3,10,20,0,false);
    eval_mha(1,4,2,3,10,20,0,true);
    return 0;
}