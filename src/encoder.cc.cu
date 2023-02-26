#include "encoder.h"
#include <chrono>
#include "transformerKernels.h"
#include "embKernels.h"
#include "fmha_api.h"

/**
@file
Transformer encoder, composed by gemm lib and
  custom cuda kernel function
*/

namespace lightseq {
namespace cuda {

template <OperationType OpType_>
Encoder<OpType_>::Encoder(int max_batch_size, /*int *p_d_token_id,*/
                          int *p_d_padding_mask, _DataType *p_d_output,
                          const TransformerWeight<OpType_> &tw,
                          cudaStream_t stream, cublasHandle_t hd,
                          const int *p_d_lang_id)
    : _max_batch_size(max_batch_size),
      /*_p_d_token_id(p_d_token_id),*/
      _p_d_padding_mask(p_d_padding_mask),
      _p_d_output(p_d_output),
      _p_d_lang_id(p_d_lang_id),
      _tw(tw),
      _stream(stream),
      _hd(hd),
      _p_d_src_emb_wei(tw.get_src_emb_wei()),
      _p_d_enc_wei(tw.get_enc_wei()),
      _fone((float)1.f), //  _DataType
      _fzero((float)0.f), // _DataType

      _atten_scaler((float)sqrt(1.f / tw._dim_per_head)), // _DataType
      _max_batch_dim(max_batch_size * tw._max_step * tw._hidden_size), // BS * max_seqlen * embedding size -> input data size
      _max_thread_per_block(256) {}

/**
Compute GPU memory size needed by transformer encoder,
  to see how these memory is used, checkout init_buffer() for detail
*/
template <OperationType OpType_>
long Encoder<OpType_>::compute_buffer_bytesize() {
  long sz1 = _max_batch_dim * 6 + _max_batch_dim * 2 // QKV*2, output x 2
              + _max_batch_size * _tw._head_num * _tw._max_step // soft max lse
             + _max_batch_size * _tw._head_num * _tw._max_step * _tw._max_step; // batch_size, num_heads, max_seqlen_q, max_seqlen_k --> P/S
  long sz2 = _max_batch_dim + _max_batch_size * _tw._max_step * _tw._inner_size; // FFN
  return max(sz1, sz2) * sizeof(_DataType); // 
}

/**
Init the GPU memory pointer which point to
  the memory buffer needed by encoder.
These buffer are used during custom cuda kernel function,
  find the corresponding function to see how these buffer are used
*/
template <OperationType OpType_>
void Encoder<OpType_>::init_buffer(void *pbuf) {
  // std::cout << "encoder buffer init start" << std::endl;
  _DataType *p_d_buf = reinterpret_cast<_DataType *>(pbuf);
  _p_d_qkv_projected = p_d_buf;
  _p_d_q = _p_d_qkv_projected + _max_batch_dim * 3;
  _p_d_k = _p_d_q + _max_batch_dim;
  _p_d_v = _p_d_k + _max_batch_dim;
  _p_d_c = _p_d_v + _max_batch_dim;
  _p_d_o = _p_d_c + _max_batch_size * _tw._head_num * _tw._max_step * _tw._max_step;
  _p_d_o_tmp = _p_d_o + _max_batch_dim;
  _p_softmax_lse = _p_d_o_tmp + _max_batch_dim; // _max_batch_size * _tw._head_num * _tw._max_step;

  _p_d_ffn_buf1 = p_d_buf;
  _p_d_ffn_buf2 = _p_d_ffn_buf1 + _max_batch_dim;
  
  // encoder and decoder use the same buffer to save gpu memory useage
  // std::cout << "encoder buffer init succeed" << std::endl;
  return;
}

/**
Some requirements needed by custom cuda kernel function
*/
template <OperationType OpType_>
std::string Encoder<OpType_>::check() {
  // if (_max_thread_per_block < _tw._hidden_size) {
  //   return "violate hidden_size <= max_thread_per_block";
  // }

  if (_tw._inner_size & 1) {
    return "violate inner_size % 2 = 0";
  }
  if (_tw._dim_per_head & 1) {
    return "violate dim_per_head % 2 = 0";
  }
  if (_tw._multilg_type == 0 && _p_d_src_emb_wei.size() != 4) {
    return "violate p_d_src_emb_wei.size() = 4";
  }
  if (_tw._multilg_type != 0 && _p_d_src_emb_wei.size() != 5) {
    return "violate p_d_src_emb_wei.size() = 5";
  }
  if (_p_d_enc_wei.size() != _tw._weight_per_enc_layer * _tw._n_enc_layer) {
    return "violate p_d_enc_wei.size() = weight_per_enc_layer * n_enc_layer";
  }
  if (_tw._multilg_type != 0 && _p_d_lang_id == nullptr) {
    return "lang id should not be null when multilg";
  }
  return "";
}

/**
Encoder inference
*/
template <OperationType OpType_>
void Encoder<OpType_>::run_one_infer(int batch_size, int batch_seq_len, int infer_mode) {

  if (batch_size > _max_batch_size) {
    throw std::runtime_error("batch size of input greater than max_batch_size");
  }
  if (batch_seq_len > _tw._max_step) {
    throw std::runtime_error("seq len of input greater than max_step"); // input seq cannot be longer than 64
  }

  /* ---step1. init--- */
  _batch_size = batch_size;
  _batch_seq_len = batch_seq_len;
  _batch_token_num = batch_size * batch_seq_len; // total token number in all batch

  int *cu_seqlens = new int[_batch_size+1];
  for(int i=0;i<=_batch_size;i++)
    cu_seqlens[i] = _batch_seq_len*i;

  CHECK_GPU_ERROR(cudaMalloc((void**)&_p_cu_seqlens, (_batch_size+1) * sizeof(int)));
  CHECK_GPU_ERROR(cudaMemcpyAsync(_p_cu_seqlens, cu_seqlens, sizeof(int) * (_batch_size+1), cudaMemcpyHostToDevice, _stream));
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
#ifdef DEBUG_RESULT_ENC
  std::cout << "batch_size-" << batch_size << " batch_seq_len-" << batch_seq_len
            << std::endl;
  print_vec(_p_d_token_id, "batch_token_ids", batch_size * batch_seq_len);
#endif

  /* ---step2. encoder feedforward, look up token embedding, add position embedding --- */
  // token_emb: _p_d_src_emb_wei[0]
  // pos_emb: _p_d_src_emb_wei[1]
  // output: _p_d_output
  // pad_mask: _p_d_padding_mask
  std::cout << "launch encoder embedding kernel" << std::endl;
  launch_enc_emb<_DataType>(_p_d_src_emb_wei[0], _p_d_src_emb_wei[1], //
                            _p_d_token_id, _p_d_output, _p_d_padding_mask, // token_ids, embedding result, generated mask 0s/1s --> _p_d_output first holds the embedding result
                            _tw._padding_id, batch_size, batch_seq_len, // batch_size=1, batch_seq_len=64
                            _tw._hidden_size, _stream, _p_d_src_emb_wei[4], // lang embedding, not used
                            _p_d_lang_id, _tw._multilg_type);

  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  CHECK_GPU_ERROR(cudaPeekAtLastError());

  print_vec(_p_d_output,"emb out", 10);
#ifdef DEBUG_RESULT_ENC
  for (int i = 0; i < _batch_size; i++) {       // batch_id
    for (int j = 0; j < _batch_seq_len; j++) {  // token_id
      std::cout << "emb out: token-" << j << std::endl;
      print_vec(_p_d_output + i * _batch_seq_len * _tw._hidden_size +
                    j * _tw._hidden_size,
                "emb out", 10);// _tw._hidden_size);
    }
  }  // not normal
  print_vec(_p_d_src_emb_wei[0], "token embedding weight", 10);
  print_vec(_p_d_src_emb_wei[1], "position embedding weight", 10);
#endif
  for (_layer_id = 0; _layer_id < _tw._n_enc_layer; _layer_id++) {
    _weight_offset = _layer_id * _tw._weight_per_enc_layer;
    if(infer_mode==0) {
      std::cout << "launch self-attention layer " << _layer_id << std::endl;
      self_attention();}
    else if(infer_mode==1) {
      std::cout << "launch flash-self-attention layer " << _layer_id << std::endl;
      flash_self_attention();
    }
#ifdef DEBUG_RESULT_ATTENTION
    /*if(_layer_id==0)*/ print_vec(_p_d_output,"encoder attention output", 10);
#endif
    // std::cout <</ "launch FFN layer " << _layer_id << std::endl;
    // ffn_add_norm();
    CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
    CHECK_GPU_ERROR(cudaPeekAtLastError());
  }

  // last layer norm
  ker_norm_layer_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_output,
      _p_d_src_emb_wei[2], _p_d_src_emb_wei[3], _max_thread_per_block);
  
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  CHECK_GPU_ERROR(cudaPeekAtLastError());

#ifdef DEBUG_RESULT_ENC
  for (int i = 0; i < _batch_size; i++) {       // batch_id
    for (int j = 0; j < _batch_seq_len; j++) {  // token_id
      std::cout << "encoder output: token-" << j << std::endl;
      print_vec(_p_d_output + i * _batch_seq_len * _tw._hidden_size + j * _tw._hidden_size, "encoder_output", 10);
    }
  }  // not normal
#endif
  return;
}

/**
Encoder self attention
*/
template <OperationType OpType_>
void Encoder<OpType_>::self_attention() {
  // print_vec(_p_d_output,"self-attention Input", 10);
  /* ---step 0. layer_norm, add output_bias to "query"--- */
  // input: _p_d_output
  // output: _p_d_q
  // scale: _p_d_enc_wei[_weight_offset]
  // bias: _p_d_enc_wei[_weight_offset + 1]
  // residual bias: _p_d_enc_wei[_weight_offset + 5]
  print_vec(_p_d_output,"encoder attention input", 10);
  ker_norm_layer_resual_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_output, _p_d_q,
      _p_d_enc_wei[_weight_offset], _p_d_enc_wei[_weight_offset + 1],
      _p_d_enc_wei[_weight_offset + 5], _max_thread_per_block, _tw._is_post_ln);

  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  FMHA_CHECK_CUDA(cudaPeekAtLastError());

  print_vec(_p_d_q,"encoder attention ker_norm_layer_resual output", 10);

  /* ---step 1. qkv = ori_q * qkv_wei + bias, and reshape qkv for multi-head
   * gemm--- */
  //  CUDA_R_32F, _AType, _BType, _CType: 0 --> real (as a float)
  //  _fone, initialized to be 1.0f
  //  _fzero, initialized to be 0.0f
  // C = alpha*op(A)*op(B) + beta*C ; alpha = _fone, A = _p_d _enc_wei[_weight_offset + 2], B = _p_d_q, beta = _fzero, C = _p_d_qkv_projected
  // A [_hidden_size * 3, _tw._hidden_size]
  // B [_tw._hidden_size, _batch_token_num]
  // C [_batch_token_num, _hidden_size * 3]
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size * 3, _batch_token_num,
      _tw._hidden_size, &_fone, _p_d_enc_wei[_weight_offset + 2], _AType,
      _tw._hidden_size * 3, _p_d_q, _BType, _tw._hidden_size, &_fzero,
      _p_d_qkv_projected, _CType, _tw._hidden_size * 3, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  // print_vec(_p_d_q,"Obtain QKV", 10);

  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  CHECK_GPU_ERROR(cudaPeekAtLastError());


  // get q, k, v by split and reshape qkv
  // original qkv: _p_d_qkv_projected
  // qkv bias: _p_d_enc_wei[_weight_offset + 3]
  // new qkv: _p_d_q
  ker_arrange_encself_qkv_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_qkv_projected,
      _p_d_enc_wei[_weight_offset + 3], _p_d_q, _max_batch_dim, _batch_seq_len,
      _tw._dim_per_head, _tw._head_num, _max_thread_per_block);

  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  CHECK_GPU_ERROR(cudaPeekAtLastError());

  // print_vec(_p_d_q,"self-attention input Q", 10);
  // print_vec(_p_d_k,"self-attention input K", 10);
  // print_vec(_p_d_v,"self-attention input V", 10);

  // Start timing
  std::chrono::high_resolution_clock::time_point timer_start;
  std::chrono::high_resolution_clock::time_point timer_end;
  timer_start = std::chrono::high_resolution_clock::now();


  /**************************************************************************************
   * Input data: Q K V dim_per_head  head_num seq_len batch_num
   * Output data: O 
   */

  /* ---step 2. correlation = q * k, perform softmax on correlation--- */
  // C+i*strideC = alpha*op(A+i*strideA)*op(B+i*strideB) + beta*(C++i*strideC), i \in 0 ... batchCount-1 ; alpha = _atten_scaler, A = _p_d_k, B = _p_d_q, beta = _fzero, C = _p_d_c
  // A [_dim_per_head, _batch_seq_len], strideA = _batch_seq_len * _dim_per_head
  // B [_batch_seq_len, _dim_per_head], strideB = _batch_seq_len * _dim_per_head
  // C [_batch_seq_len, _dim_per_head], strideC = _batch_seq_len * _batch_seq_len
  CHECK_GPU_ERROR(cublasGemmStridedBatchedEx(
      _hd, CUBLAS_OP_T, CUBLAS_OP_N, _batch_seq_len, _batch_seq_len,
      _tw._dim_per_head, &_atten_scaler, _p_d_k, _AType, _tw._dim_per_head,
      _batch_seq_len * _tw._dim_per_head, _p_d_q, _BType, _tw._dim_per_head,
      _batch_seq_len * _tw._dim_per_head, &_fzero, _p_d_c, _CType,
      _batch_seq_len, _batch_seq_len * _batch_seq_len,
      _batch_size * _tw._head_num, CUDA_R_32F, // matmul by head, independent batch and head
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  CHECK_GPU_ERROR(cudaPeekAtLastError());

  // print_vec(_p_d_c,"self-attention softmax output S", 10);

  // if(_layer_id==0)
  // {print_vec(_p_d_k,"Attention first GEMM inputA", 10);  // 4096
  // print_vec(_p_d_q,"Attention first GEMM inputB", 10); // 4096

  // print_vec(_p_d_c,"Attention first GEMM output S[0][0-3]", 4); // 4096

  // _p_d_c: correlation
  // _p_d_padding_mask: src_padding_mask
  ker_correlation_softmax_encself_launcher<_DataType>(
      _batch_size, _batch_seq_len, _tw._head_num, _stream, _p_d_c,
      _p_d_padding_mask);
  

  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  CHECK_GPU_ERROR(cudaPeekAtLastError());
  // print_vec(_p_d_c, "Attention softmax output P[0][0-3]", 4); // 4096

  /* ---step 3. new_q = correlation * v--- */
  // C+i*strideC = alpha*op(A+i*strideA)*op(B+i*strideB) + beta*(C+i*strideC), i \in 0 ... batchCount-1 ; alpha = _fone, A = _p_d_v, B = _p_d_c, beta = _fzero, C = _p_d_q
  CHECK_GPU_ERROR(cublasGemmStridedBatchedEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._dim_per_head, _batch_seq_len,
      _batch_seq_len, &_fone, _p_d_v, _AType, _tw._dim_per_head,
      _batch_seq_len * _tw._dim_per_head, _p_d_c, _BType, _batch_seq_len,
      _batch_seq_len * _batch_seq_len, &_fzero, _p_d_q, _CType,
      _tw._dim_per_head, _batch_seq_len * _tw._dim_per_head,
      _batch_size * _tw._head_num, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  CHECK_GPU_ERROR(cudaPeekAtLastError());
  // print_vec(_p_d_q,"Attention second GEMM output O[0][0-3]", 128); // 4096
  // End timing
  timer_end = std::chrono::high_resolution_clock::now();
  std::cout << "cuBLAS GEMM non-fused Self-Attention: " << std::chrono::duration_cast<std::chrono::microseconds>(timer_end - timer_start).count() << " microseconds" << std::endl;

  /* reshape Scaled Dot-Product Attention output 
      use v to save reshaped q, since they are in same size and v will not be use again before the next multi-head-attention
  */
  // 4 -> 2 dim 
  // _p_d_q: ori_q -> ori output of attention in 4d
  // _p_d_v: new_q -> new output of attention in 2d
  ker_arrange_atten_output_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_q, _p_d_v,
      _batch_seq_len, _tw._dim_per_head, _tw._head_num, _max_thread_per_block);

  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  CHECK_GPU_ERROR(cudaPeekAtLastError());

  // if(_layer_id==1) print_vec(_p_d_v,"ker_arrange_atten_output_launcher output b 0", 512);
  // if(_layer_id==1) print_vec(_p_d_v+64*512,"ker_arrange_atten_output_launcher output b 1", 512);

  /* ---step 4. feed forward layer, new_q = ori_q + new_q * output_wei--- */
  // C = alpha*op(A)*op(B) + beta*C ; alpha = _fone, A = _p_d_enc_wei[_weight_offset + 4], B = _p_d_v, beta = _fone, C = _p_d_output
  // CHECK_GPU_ERROR(cublasGemmEx(
  //     _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size, _batch_token_num,
  //     _tw._hidden_size, &_fone, _p_d_enc_wei[_weight_offset + 4], _AType,
  //     _tw._hidden_size, _p_d_v, _BType, _tw._hidden_size, &_fone, _p_d_output,
  //     _CType, _tw._hidden_size, _computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CHECK_GPU_ERROR(cudaMemcpy(_p_d_output, _p_d_v, sizeof(_DataType) * _batch_size * _batch_seq_len * _tw._dim_per_head * _tw._head_num, cudaMemcpyDeviceToDevice));
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  CHECK_GPU_ERROR(cudaPeekAtLastError());
  return;
}


/************************************************************************************************
*************************************************************************************************
*************************************************************************************************
Encoder self attention (FLASH-Attention Implementation)
*************************************************************************************************
*************************************************************************************************
*************************************************************************************************/

template <OperationType OpType_>
void Encoder<OpType_>::flash_self_attention() {
  // print_vec(_p_d_output,"self-attention Input", 10);
  /* %%%%% ---step 0. %%%%%% --- */
  /*  layer_norm, add output_bias to "query"--- */
  // input: _p_d_output
  // output: _p_d_q
  // scale: _p_d_enc_wei[_weight_offset]
  // bias: _p_d_enc_wei[_weight_offset + 1]
  // residual bias: _p_d_enc_wei[_weight_offset + 5]
  // std::cout << "launch layer norm and residual bias" << std::endl;
  print_vec(_p_d_output,"encoder attention input", 10);
  ker_norm_layer_resual_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_output, _p_d_q,
      _p_d_enc_wei[_weight_offset], _p_d_enc_wei[_weight_offset + 1],
      _p_d_enc_wei[_weight_offset + 5], _max_thread_per_block, _tw._is_post_ln);
  
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  CHECK_GPU_ERROR(cudaPeekAtLastError());

  print_vec(_p_d_q,"encoder attention ker_norm_layer_resual output", 10);

  // CHECK_GPU_ERROR(cudaMemcpy(_p_d_q, _p_d_output, sizeof(_DataType) * _batch_size * _batch_seq_len * _tw._dim_per_head * _tw._head_num, cudaMemcpyDeviceToDevice));

  /* --- qkv = ori_q * qkv_wei + bias, and reshape qkv for multi-head  gemm--- */
  //  CUDA_R_32F, _AType, _BType, _CType: 0 --> real (as a float)
  //  _fone, initialized to be 1.0f
  //  _fzero, initialized to be 0.0f
  // C = alpha*op(A)*op(B) + beta*C ; alpha = _fone, A = _p_d _enc_wei[_weight_offset + 2], B = _p_d_q, beta = _fzero, C = _p_d_qkv_projected
  // A [_hidden_size * 3, _hidden_size]
  // B [_hidden_size, _batch_token_num]
  // C [_hidden_size * 3, _batch_token_num]
  // std::cout << "launch GEMM for generate QKV" << std::endl;
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size * 3, _batch_token_num,
      _tw._hidden_size, &_fone, _p_d_enc_wei[_weight_offset + 2], _AType,
      _tw._hidden_size * 3, _p_d_q, _BType, _tw._hidden_size, &_fzero,
      _p_d_qkv_projected, _CType, _tw._hidden_size * 3, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  CHECK_GPU_ERROR(cudaPeekAtLastError());

  /* %%%%% --- step 1. %%%%%% --- */

  // get q, k, v by split and reshape qkv
  // original qkv: _p_d_qkv_projected
  // qkv bias: _p_d_enc_wei[_weight_offset + 3]
  // new qkv: _p_d_q
  // std::cout << "launch split and reshape QKV" << std::endl;
  ker_arrange_encself_qkv_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_qkv_projected,
      _p_d_enc_wei[_weight_offset + 3], _p_d_q, _max_batch_dim, _batch_seq_len,
      _tw._dim_per_head, _tw._head_num, _max_thread_per_block);

  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  CHECK_GPU_ERROR(cudaPeekAtLastError());
  /* reshape input
  */
  // 4 -> 2 dim 
  // _p_d_q: ori_q -> ori output of attention in 4d
  // _p_d_v: new_q -> new output of attention in 2d
  // std::cout << "launch reshape Scaled Dot-Product Attention output" << std::endl;
  ker_arrange_atten_output_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_q, _p_d_q, _batch_seq_len, _tw._dim_per_head, _tw._head_num, _max_thread_per_block);
  
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  CHECK_GPU_ERROR(cudaPeekAtLastError());

  ker_arrange_atten_output_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_k, _p_d_k, _batch_seq_len, _tw._dim_per_head, _tw._head_num, _max_thread_per_block);

  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  CHECK_GPU_ERROR(cudaPeekAtLastError());

  ker_arrange_atten_output_launcher<_DataType>(
    _batch_token_num, _tw._hidden_size, _stream, _p_d_v, _p_d_v, _batch_seq_len, _tw._dim_per_head, _tw._head_num, _max_thread_per_block);

  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  CHECK_GPU_ERROR(cudaPeekAtLastError());
  // print_vec(_p_d_q,"self-attention input Q", 10);
  // print_vec(_p_d_k,"self-attention input K", 10);
  // print_vec(_p_d_v,"self-attention input V", 10);

  // self attention: max_seqlen_q_ == max_seqlen_k_
  const int max_seqlen_q_ = _tw._max_step; // 
  const int max_seqlen_k_ = _tw._max_step; 
  const float softmax_scale = _atten_scaler;

  const int batch_size = _batch_size;
  const int total_q = _batch_token_num; // 0: valid tokens (<batch*seq_len)
  const int num_heads = _tw._head_num; // 1: nhead
  const int head_size = _tw._dim_per_head; //2: head dim

  // TORCH.TENSOR.STRIDE
  const uint32_t q_stride_0 = num_heads*head_size;
  const uint32_t k_stride_0 = num_heads*head_size; 
  const uint32_t v_stride_0 = num_heads*head_size;
  const uint32_t q_stride_1 = head_size;
  const uint32_t k_stride_1 = head_size;
  const uint32_t v_stride_1 = head_size;
  

  // std::cout << "max_seqlen_q_ : " << max_seqlen_q_ << std::endl;
  // std::cout << "max_seqlen_k_ : " << max_seqlen_k_ << std::endl;
  // std::cout << "softmax_scale : " << softmax_scale << std::endl;
  // std::cout << "batch_size : " << batch_size << std::endl;
  // std::cout << "total_q : " << total_q << std::endl;
  // std::cout << "q_stride_0 : " << q_stride_0 << std::endl;
  // std::cout << "k_stride_0 : " << k_stride_0 << std::endl;
  // std::cout << "v_stride_0 : " << v_stride_0 << std::endl;
  // std::cout << "q_stride_1 : " << q_stride_1 << std::endl;
  // std::cout << "k_stride_1 : " << k_stride_1 << std::endl;
  // std::cout << "v_stride_1 : " << v_stride_1 << std::endl;

  int blocksize_c = 64; // 128 / 256
  // Need to round max_seqlen_k to multiples of blocksize_c
  const int max_seqlen_k = ((max_seqlen_k_ + blocksize_c - 1) / blocksize_c) * blocksize_c;
  const int max_seqlen_q = ((max_seqlen_q_ + 16 - 1) / 16) * 16; // 16 is STEP


  if (max_seqlen_q != 64) {
    throw std::runtime_error("max_seqlen_q != 64");
  }
  if (max_seqlen_k != 64) {
    throw std::runtime_error("max_seqlen_k != 64");
  }

  // assume max_seqlen_k > blocksize_c
  // auto dprops = at::cuda::getCurrentDeviceProperties();  fmha::
  Launch_params<FMHA_fprop_params> launch_params(/*dprops,*/ _stream);
  fmha::set_params_fprop(launch_params.params,
                     batch_size, // b
                     max_seqlen_q,  
                     max_seqlen_k,
                     num_heads, // h
                     head_size, // d
                     _p_d_q, _p_d_k, _p_d_v,
                     q_stride_0,k_stride_0,v_stride_0,
                     q_stride_1,k_stride_1,v_stride_1,
                     _p_cu_seqlens,
                     _p_cu_seqlens,
                     _p_d_o,
                     _p_d_o_tmp,
                     _p_d_c, // S=softmax(Q*K^T)
                     _p_softmax_lse, // l(x)
                     softmax_scale);

  // Start timing
  std::chrono::high_resolution_clock::time_point timer_start;
  std::chrono::high_resolution_clock::time_point timer_end;
  timer_start = std::chrono::high_resolution_clock::now();

  using elem_type = lightseq::cuda::OperationTypeTraits<lightseq::cuda::OperationType::FP16>::DataType;
  // d, seqlen_k,
  // std::cout << "launch fmha_fp16 for d=64, seqlen=64" << std::endl;
  // outer loop step: 64
  // dim: 64
  // inner STEP: 16
  using Kernel_traits = FMHA_kernel_traits<64, 64, 16, 1, 4, 0x08u, elem_type>;
  fmha::run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params);

   // End timing
  timer_end = std::chrono::high_resolution_clock::now();
  std::cout << "fused Flash-Attention: " << std::chrono::duration_cast<std::chrono::microseconds>(timer_end - timer_start).count() << " microseconds" << std::endl;

  // print_vec(_p_d_c,"self-attention softmax output S", 10);

  CHECK_GPU_ERROR(cudaMemcpy(_p_d_v, _p_d_o, sizeof(_DataType) * _batch_size * _batch_seq_len * _tw._dim_per_head * _tw._head_num, cudaMemcpyDeviceToDevice));
  // if(_layer_id==1) print_vec(_p_d_v,"ker_arrange_atten_output_launcher output b 0", 512);
  // if(_layer_id==1) print_vec(_p_d_v+64*512,"ker_arrange_atten_output_launcher output b 1", 512);
  // print_vec(_p_d_v,"ker_arrange_atten_output_launcher output", 10);
  // print_vec(_p_d_v,"self-attention output V", 10);

  /* ---step 4. feed forward layer, new_q = ori_q + new_q * output_wei--- */
  // C = alpha*op(A)*op(B) + beta*C ; alpha = _fone, A = _p_d_enc_wei[_weight_offset + 4], B = _p_d_v, beta = _fone, C = _p_d_output
  // std::cout << "launch GEMM for FFN" << std::endl;
  // [_tw._hidden_size,_tw._hidden_size]*[_tw._hidden_size,_batch_token_num] -> [_batch_token_num,_tw._hidden_size]
  // CHECK_GPU_ERROR(cublasGemmEx(
  //     _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size, _batch_token_num,
  //     _tw._hidden_size, &_fone, _p_d_enc_wei[_weight_offset + 4], _AType,
  //     _tw._hidden_size, _p_d_v, _BType, _tw._hidden_size, &_fone, _p_d_output,
  //     _CType, _tw._hidden_size, _computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CHECK_GPU_ERROR(cudaMemcpy(_p_d_output, _p_d_v, sizeof(_DataType) * _batch_size * _batch_seq_len * _tw._dim_per_head * _tw._head_num, cudaMemcpyDeviceToDevice));
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  CHECK_GPU_ERROR(cudaPeekAtLastError());
  return;
}



template <OperationType OpType_>
void Encoder<OpType_>::ffn_add_norm() {
  /* ---step 0. layer_norm, add output_bias to "query"--- */
  ker_norm_layer_resual_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_output, _p_d_ffn_buf1,
      _p_d_enc_wei[_weight_offset + 6], _p_d_enc_wei[_weight_offset + 7],
      _p_d_enc_wei[_weight_offset + 11], _max_thread_per_block,
      _tw._is_post_ln);
  /* ---step 1. first ffn layer--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._inner_size, _batch_token_num,
      _tw._hidden_size, &_fone, _p_d_enc_wei[_weight_offset + 8], _AType,
      _tw._inner_size, _p_d_ffn_buf1, _BType, _tw._hidden_size, &_fzero,
      _p_d_ffn_buf2, _CType, _tw._inner_size, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  if (_tw._use_gelu) {
    ker_bias_gelu_launcher<_DataType>(
        _batch_token_num, _max_thread_per_block, _stream, _p_d_ffn_buf2,
        _p_d_enc_wei[_weight_offset + 9], _tw._inner_size);
  } else {
    ker_bias_relu_launcher<_DataType>(
        _batch_token_num, _max_thread_per_block, _stream, _p_d_ffn_buf2,
        _p_d_enc_wei[_weight_offset + 9], _tw._inner_size);
  }
  /* ---step 2. second ffn layer--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size, _batch_token_num,
      _tw._inner_size, &_fone, _p_d_enc_wei[_weight_offset + 10], _AType,
      _tw._hidden_size, _p_d_ffn_buf2, _BType, _tw._inner_size, &_fone,
      _p_d_output, _CType, _tw._hidden_size, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  return;
}

template class Encoder<OperationType::FP16>;
// template class Encoder<OperationType::FP32>;

}  // namespace cuda
}  // namespace lightseq
