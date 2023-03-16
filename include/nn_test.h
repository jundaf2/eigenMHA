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
    std::function<bool(float,float,float)> NEAR2 = [](float a, float b, float prec) -> bool { return ((a != a && b != b) 
      || (a == std::numeric_limits<typename std::remove_reference<decltype(a)>::type>::infinity() 
        && b == std::numeric_limits<typename std::remove_reference<  decltype(b)>::type>::infinity()) 
      || (-a == std::numeric_limits<typename std::remove_reference< decltype(a)>::type>::infinity() 
        && -b == std::numeric_limits<typename std::remove_reference<  decltype(b)>::type>::infinity()) 
      || (abs(a - b) / abs(a) < prec) || (abs(a - b) / abs(b) < prec) || (abs(a - b)*1e6 < prec)); };

    std::function<void(bool, std::string, int)> ASSERT = [](bool cond, std::string info, int line) -> void { if(!(cond)){ std::string s = info + " @ " + __FILE__ + " (" + std::to_string(line) + ")"; printf(ANSI_COLOR_BLUE "ASSERT FAILED:: %s\n" ANSI_COLOR_RESET, s.c_str()); } };

    
    std::mt19937 rng; //{std::random_device{}()};
    std::uniform_real_distribution<float> uf_distribution;

  protected:
    json torch_test_data_bank;
    json raw_test_data_bank;
    json input_data_bank;

    void register_torch_test_data(const torch::Tensor& x, std::string name){
      torch::Tensor x_c = x.to(torch::kCPU).contiguous();
      std::vector<float> x_v(x_c.data_ptr<float>(), x_c.data_ptr<float>() + x_c.numel());
      torch_test_data_bank[name] = x_v;  
    }

    void register_raw_test_data(const float* x, size_t len, std::string name){
      std::vector<float> x_v(x, x + len);
      raw_test_data_bank[name] = x_v;
    }

    void set_input_vec(const float* x, size_t len, std::string name){
      std::vector<float> x_v(x, x + len);
      input_data_bank[name] = x_v;
    }

    std::vector<float> get_input_vec(std::string name){
      std::vector<float> vec (input_data_bank[name].get<std::vector<float>>());
      return vec;
    }

    void get_input_ten(torch::Tensor& ten, std::string name, torch::TensorOptions options){
      for(size_t i=0; i<ten.numel(); i++)
        ten.data_ptr<float>()[i] = this->get_input_vec(name).data()[i];
    }

    void set_random(float rand_min, float rand_max, unsigned int random_seed){
      rng = std::mt19937(random_seed);
      uf_distribution = std::uniform_real_distribution<float>(rand_min, rand_max);
    }

    std::vector<float> gen_input_vec(size_t len){
      std::vector<float> vec(len);
      std::generate(std::begin(vec), std::end(vec), [&]{return uf_distribution(rng);} ); // 1
      return vec;
    }

    void print_vec(const std::vector<float> outv, std::string outn, int start, int end) {
      std::cout << outn << ": ";
      std::copy(outv.begin() + start, outv.begin() + end, std::ostream_iterator<float>(std::cout, ", "));
      std::cout << std::endl;
    }

    std::string print_vec(const std::vector<float> outv, int start, int end) {
      std::stringstream ss;
      std::copy(outv.begin() + start, outv.begin() + (outv.size()>end?end:outv.size()), std::ostream_iterator<float>(ss, ", "));
      ss << std::endl;
      return ss.str();
    }

    void print_ten(const torch::Tensor& x, std::string name){
      std::cout << name << ": " << x << std::endl;
    }

    void print_json_bank(json j, std::string bank_name){
      for (auto it = j.begin(); it != j.end(); ++it) {
        fmt::print("{}  {}  =  {}\n", bank_name,  it.key(), it.value());
      }
    }

  public:
    virtual void init_data() = 0;
    virtual void run_torch_dnn() = 0;
    virtual void run_my_dnn() = 0;

    void verify() {
      ASSERT(torch_test_data_bank.size() == raw_test_data_bank.size(), "test data and verify data should have the same number of entries", __LINE__);

      std::string error_name_list;
      for(auto& [name, data_vec] : raw_test_data_bank.items()) {
        if(!torch_test_data_bank.contains(name)){
          error_name_list += "  ";
          error_name_list += name;
          error_name_list += "\n";
        }
      }
      ASSERT(error_name_list.empty(), "verify data and test data should have the same set of entries", __LINE__);

      // print_json_bank(input_data_bank, "input_data_bank");

        // INFO("raw_test_data_bank: \n" << raw_test_data_bank.dump());
        // INFO("torch_test_data_bank: \n" << torch_test_data_bank.dump());
      for(auto& [name, data_vec] : raw_test_data_bank.items()){
        SECTION("The results must match for " + (ANSI_COLOR_CYAN + name + ANSI_COLOR_RESET)) {
          bool is_near2 = true;
          for (int i = 0; i < data_vec.size(); i++) {
            is_near2 &= NEAR2(data_vec[i], torch_test_data_bank[name][i], 1e-3);
          }
          INFO(ANSI_COLOR_RED "* Your DNN: " ANSI_COLOR_RESET << print_vec(data_vec,0,64) << ANSI_COLOR_GREEN " \n * Torch DNN: " ANSI_COLOR_RESET << print_vec(torch_test_data_bank[name],0,64));
          CHECK(is_near2);
        }
      }      
    }
};

}
