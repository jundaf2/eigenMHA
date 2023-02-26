#include "model_base.h"
#include "util.h"

/**
@file
Example of how to run transformer inference using our implementation.
*/

int main(int argc, char* argv[]) {
  std::string model_weights_path = argv[1];

  std::vector<int> example_input; // = {88,  74, 10, 2057, 362, 9,    284, 6};
  for(int i=0;i<128;i++){
    example_input.push_back(i);
  }
  int eg_seq_len = example_input.size();
  int max_batch_size = 32; //32
  int batch_seq_len = eg_seq_len;

  const int infer_mode = atoi(argv[2]); // 0 is normal, 1 is flash
  const int batch_size = atoi(argv[3]); // number of batch
  if (batch_size > max_batch_size) {
    throw std::runtime_error("batch_size exceeds the maximum (" + std::to_string(max_batch_size) + ")!");
  }

  std::vector<int> host_input;
  // host_input -> [32,10]
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < batch_seq_len; ++j) {
      host_input.push_back(example_input[j % eg_seq_len]);
    }
  }

  auto model = lightseq::cuda::LSModelFactory::GetInstance().CreateModel("Transformer", model_weights_path, max_batch_size);
  CHECK_GPU_ERROR(cudaSetDevice(0));

  void* d_input;
  CHECK_GPU_ERROR(
      cudaMalloc(&d_input, sizeof(int) * batch_size * batch_seq_len)); // real input --> token ids
  CHECK_GPU_ERROR(cudaMemcpy(
      d_input, host_input.data(), sizeof(int) * batch_size * batch_seq_len,
      cudaMemcpyHostToDevice));

  model->set_input_ptr(0, d_input); // token ids of the encoder (src lg)
  model->set_input_shape(0, {batch_size, batch_seq_len}); // shape of real input seq of token ids
  
  for (int i = 0; i < model->get_output_size(); i++) { // alloacte the output mem for 2 output data
    void* d_output;
    std::vector<int> shape = model->get_output_max_shape(i); 
    // shape of target_ids: max BS x beam size x embedding size  [128,4,512]
    // shape of target_scores: max BS x beam size   [128,4]
    int total_size = 1;
    for (int j = 0; j < shape.size(); j++) {
      total_size *= shape[j];
    }
    CHECK_GPU_ERROR(
        cudaMalloc(&d_output, total_size * sizeof(int))); // target_ids is token id (int), target_scores is score (float, share the same size as int)
    model->set_output_ptr(i, d_output);
  }

  std::cout << "infer preprocessing finished" << std::endl;

  /* ---step5. infer and log--- */
  for (int i = 0; i < 1; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    model->Infer(infer_mode);
    lightseq::print_time_duration(start, "one infer time", 0);
  }

  for (int i = 0; i < model->get_output_size(); i++) { // 2
    const void* d_output;
    d_output = static_cast<const float*>(model->get_output_ptr(i));
    std::vector<int> shape = model->get_output_shape(i);
    std::cout << "Transformer output shape: ";
    int total_size = 1;
    for (int j = 0; j < shape.size(); j++) {
      std::cout << shape[j] << " ";
      total_size *= shape[j];
    }
    std::cout << std::endl;

    // print_vec((int*)d_output, "Transformer output", total_size); // copy from device to host, print several values from beginning
  }

  // const int* res = model.get_result_ptr();
  // const float* res_score = model.get_score_ptr();
  // print_vec(res_score, "res score", 5);
  return 0;
}
