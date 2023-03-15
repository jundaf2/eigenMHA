#pragma once
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
#include "torch/torch.h"


namespace dnn_test {
class Module {
  using json = nlohmann::json;

  protected:
    json torch_tensor_bank;
    json raw_data_bank;

    void add_torch_data(torch::Tensor x, std::string name){
      torch::Tensor x_c = x.contiguous();
      std::vector<float> x_v(x_c.data_ptr<float>(), x_c.data_ptr<float>() + x_c.numel());
      torch_tensor_bank[name] = x_v;
    }

    void add_raw_data(const float* x, size_t len, std::string name){
      std::vector<float> x_v(x, x + len);
      raw_data_bank[name] = x_v;
    }
  public:
    virtual void run_dnn_torch();
    virtual void run_dnn();

    void verify() {
      INFO("Verifying");

      SECTION("The test data and verify data should have the same number of entries ...") {
        REQUIRE(torch_tensor_bank.size() == raw_data_bank.size());
      }

      auto NEAR2 = [](int a, int b, float prec) -> bool { ((a != a && b != b) 
      || (a == std::numeric_limits<typename std::remove_reference<decltype(a)>::type>::infinity() 
        && b == std::numeric_limits<typename std::remove_reference<  decltype(b)>::type>::infinity()) 
      || (-a == std::numeric_limits<typename std::remove_reference< decltype(a)>::type>::infinity() 
        && -b == std::numeric_limits<typename std::remove_reference<  decltype(b)>::type>::infinity()) 
      || (abs(a - b) / abs(a) < prec) || (abs(a - b) / abs(b) < prec) || (abs(a - b) < prec)); };

      for(auto& [name, data_vec] : raw_data_bank.items()){

        SECTION("Verify data must contains the entries that test data has ...") {
          REQUIRE(torch_tensor_bank.contains(name));
        }

        SECTION("The results must match") {
          for (int i = 0; i < data_vec.size(); i++) {
            SECTION("The results did not match at index " + std::to_string(i)) {
              REQUIRE(NEAR2(data_vec[i], torch_tensor_bank[name][i], 1e-3));
            }
          }
        }
      }      
    }
};

}
