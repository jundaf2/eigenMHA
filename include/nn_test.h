#pragma once
#define CATCH_CONFIG_CPP11_TO_STRING
#define CATCH_CONFIG_COLOUR_ANSI
#include <functional>
#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <map>
#include "catch.hpp"
#include "fmt.hpp"
#include "utils.hpp"
#include "clara.hpp"
#include "torch/torch.h"


namespace nn_test {
class nnTest {
  private:
    std::function<bool(float,float,float)> NEAR2 = [](float a, float b, float prec) -> bool { return ((a != a && b != b) 
      || (a == std::numeric_limits<typename std::remove_reference<decltype(a)>::type>::infinity() 
        && b == std::numeric_limits<typename std::remove_reference<  decltype(b)>::type>::infinity()) 
      || (-a == std::numeric_limits<typename std::remove_reference< decltype(a)>::type>::infinity() 
        && -b == std::numeric_limits<typename std::remove_reference<  decltype(b)>::type>::infinity()) 
      || (abs(a - b) / abs(a) < prec) || (abs(a - b) / abs(b) < prec) || (abs(a - b) < prec)); };

    std::function<void(bool, std::string, int)> ASSERT = [](bool cond, std::string info, int line) -> void { if(!(cond)){ std::string s = info + " @ " + __FILE__ + " (" + std::to_string(line) + ")"; printf(ANSI_COLOR_BLUE "ASSERT FAILED:: %s\n" ANSI_COLOR_RESET, s.c_str()); } };

    
    std::mt19937 rng; //{std::random_device{}()};
    std::uniform_real_distribution<float> uf_distribution;

    int print_el_num = 64;

  protected:
    std::map<std::string,std::vector<float>> torch_test_data_bank;
    std::map<std::string,std::vector<float>> raw_test_data_bank;
    std::map<std::string,std::vector<float>> input_data_bank;


    void register_torch_test_data(const torch::Tensor& x, std::string name){
      torch::Tensor x_c = x.to(torch::kCPU).contiguous();
      torch_test_data_bank.emplace(name, std::vector<float>(x_c.data_ptr<float>(), x_c.data_ptr<float>() + x_c.numel()));
    }

    void register_raw_test_data(const float* x, size_t len, std::string name){
      raw_test_data_bank.emplace(name, std::vector<float>(x, x + len));
    }

    void set_input_vec(const float* x, size_t len, std::string name){
      input_data_bank.emplace(name, std::vector<float>(x, x + len));
    }

    std::vector<float> get_input_vec(std::string name){
      std::vector<float> vec;
      if (input_data_bank.find(name) != input_data_bank.end()){
        vec = input_data_bank[name];
      }
      else{
        std::cout << ANSI_COLOR_RED << name << " not found" << ANSI_COLOR_RESET << std::endl;
      }
      return vec;
    }

    void get_input_ten(torch::Tensor& ten, std::string name, torch::TensorOptions options){
      for(size_t i=0; i<ten.numel(); i++)
        ten.data_ptr<float>()[i] = this->get_input_vec(name).data()[i];
    }

    void set_random_seed(unsigned int random_seed){
      rng = std::mt19937(random_seed);
    }

    std::vector<float> gen_rand_input(float rand_min, float rand_max, size_t len){
      
      uf_distribution = std::uniform_real_distribution<float>(rand_min, rand_max);

      std::vector<float> vec(len);
      std::generate(std::begin(vec), std::end(vec), [&]{return uf_distribution(rng);} ); 
      return vec;
    }

    std::vector<float> gen_constant_input(float c, size_t len){
      std::vector<float> vec(len);
      std::generate(std::begin(vec), std::end(vec), [&]{return c;} ); 
      return vec;
    }

    void print_vec(const std::vector<float> outv, std::string outn, int start) {
      std::cout << outn << ": ";
      std::copy(outv.begin() + start, outv.begin() + start + print_el_num, std::ostream_iterator<float>(std::cout, ", "));
      std::cout << std::endl;
    }

    std::string print_str_vec(const std::vector<float> outv, int start) {
      std::stringstream ss;
      std::ostream_iterator<float> sout(ss, ", ");
      std::copy(outv.begin() + start, outv.begin() + (outv.size()>(print_el_num+start)?(print_el_num+start):outv.size()), sout);
      ss << std::endl;
      return ss.str();
    }

    void print_ten(const torch::Tensor& x, std::string name){
      std::cout << name << ": " << x << std::endl;
    }

    void print_data_bank(std::map<std::string,std::vector<float>>& data_bank, std::string bank_name){
      for (auto it = data_bank.begin(); it != data_bank.end(); ++it) {
        fmt::print(std::string(ANSI_COLOR_CYAN) + "** {}  {}" + ANSI_COLOR_RESET + "  =  {}\n", bank_name,  it->first, it->second);
      }
    }

  public:
    virtual void init_data() = 0;
    virtual void run_torch_dnn() = 0;
    virtual void run_my_dnn() = 0;

    void set_print_el_num(int n) {print_el_num = n;}

    void verify() {
      ASSERT(torch_test_data_bank.size() == raw_test_data_bank.size(), "test data and verify data should have the same number of entries", __LINE__);

      std::string error_name_list;
      for(auto& [name, data_vec] : raw_test_data_bank) {
        if(torch_test_data_bank.find(name)==torch_test_data_bank.end()){
          error_name_list += "  ";
          error_name_list += name;
          error_name_list += "\n";
        }
      }
      ASSERT(error_name_list.empty(), "verify data and test data should have the same set of entries", __LINE__);

      // print_data_bank(raw_test_data_bank, "raw_test_data_bank");
      // print_data_bank(torch_test_data_bank, "torch_test_data_bank");
      
      for(auto it = raw_test_data_bank.begin(); it != raw_test_data_bank.end(); ++it){
        std::string name = it->first;
        std::vector<float> data_vec = it->second;
        SECTION(std::string(ANSI_COLOR_RED) + name + ANSI_COLOR_RESET) {
          bool is_near2 = true;
          for (int i = 0; i < data_vec.size(); i++) {
            bool is_this_near2 = NEAR2(data_vec[i], torch_test_data_bank[name][i], 1e-1);
            if(!is_this_near2){
              fmt::print(ANSI_COLOR_RED "ERROR @ {}[{}] {} vs {}\n" ANSI_COLOR_RESET, name, i, data_vec[i], torch_test_data_bank[name][i]);
            }
            is_near2 &= is_this_near2;
          }
          INFO(ANSI_COLOR_RED "* Your DNN: " ANSI_COLOR_RESET << print_str_vec(data_vec,0) << ANSI_COLOR_GREEN " \n * Torch DNN: " ANSI_COLOR_RESET << print_str_vec(torch_test_data_bank[name],0));
          CHECK(is_near2);
        }
      }      
    }
};

}
