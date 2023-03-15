#pragma once
#define CATCH_CONFIG_CPP11_TO_STRING
#define CATCH_CONFIG_COLOUR_ANSI
#define CATCH_CONFIG_MAIN
#include <functional>
#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include "catch.hpp"
#include "fmt.hpp"
#include "utils.hpp"
#include "json.hpp"
#include "clara.hpp"
#include "torch/torch.h"


namespace nn_test {
class nnTest {
  using json = nlohmann::json;
  private:
    std::function<bool(int,int,float)> NEAR2 = [](int a, int b, float prec) -> bool { ((a != a && b != b) 
      || (a == std::numeric_limits<typename std::remove_reference<decltype(a)>::type>::infinity() 
        && b == std::numeric_limits<typename std::remove_reference<  decltype(b)>::type>::infinity()) 
      || (-a == std::numeric_limits<typename std::remove_reference< decltype(a)>::type>::infinity() 
        && -b == std::numeric_limits<typename std::remove_reference<  decltype(b)>::type>::infinity()) 
      || (abs(a - b) / abs(a) < prec) || (abs(a - b) / abs(b) < prec) || (abs(a - b) < prec)); };
  protected:
    json torch_test_data_bank;
    json raw_test_data_bank;
    json input_data_bank;

    void register_torch_test_data(torch::Tensor x, std::string name){
      torch::Tensor x_c = x.contiguous();
      std::vector<float> x_v(x_c.data_ptr<float>(), x_c.data_ptr<float>() + x_c.numel());
      torch_test_data_bank[name] = x_v;
    }

    void register_raw_test_data(const float* x, size_t len, std::string name){
      std::vector<float> x_v(x, x + len);
      raw_test_data_bank[name] = x_v;
    }

    void set_input_data(const float* x, size_t len, std::string name){
      std::vector<float> x_v(x, x + len);
      input_data_bank[name] = x_v;
    }

    std::vector<float> get_input_data(std::string name){
      return input_data_bank[name];
    }

    torch::Tensor get_input_tensor(std::string name, torch::IntArrayRef shape, torch::TensorOptions options){
      return torch::from_blob(this->get_input_data(name).data(), shape, options);
    }

    std::vector<float> gen_input_data(size_t len, unsigned int random_seed){
      std::mt19937 rng {random_seed}; //{std::random_device{}()};
      std::uniform_real_distribution<float> uf_distribution(-1.0f, 1.0f);
      std::vector<float> vec(len);
      std::generate(begin(vec), end(vec), [&]{return uf_distribution(rng);} );
      return vec;
    }

  public:
    virtual void init_data() = 0;
    virtual void run_torch_dnn() = 0;
    virtual void run_my_dnn() = 0;

    void verify() {
      std::cout << raw_test_data_bank.dump() << std::endl;
      // std::cout << torch_test_data_bank.dump() << std::endl;
      SECTION("test data and verify data should have the same number of entries") {
        REQUIRE(torch_test_data_bank.size() == raw_test_data_bank.size());
      }

      for(auto& [name, data_vec] : raw_test_data_bank.items()){

        SECTION("verify data and test data different entry: "+name) {
          REQUIRE(torch_test_data_bank.contains(name));
        }

        SECTION("The results must match") {
          for (int i = 0; i < data_vec.size(); i++) {
            INFO("The results of " << name << " does not match at index " << i << ". your DNN data: " << data_vec[i] << " the torch data:" << torch_test_data_bank[name][i]); 
            REQUIRE(NEAR2(data_vec[i], torch_test_data_bank[name][i], 1e-3));
          }
        }
      }      
    }
};

}
