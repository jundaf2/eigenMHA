#include <Eigen/Core>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace Eigen;

void SoftMax(Eigen::MatrixXf mat)
{
	MatrixXf m = mat.array().exp();
	cout << "\n" << m;
	float sum = m.sum();
	cout << "\n" << sum;
	MatrixXf n = m / sum;
	cout << "\n" << n;
	cout << "\n" << n.sum();
}

int main(int argc, char* argv[]) {
  MatrixXf input(1, 4);

	input << -1, -2, -3, -4;
	cout << "input: \n" << input;
	SoftMax(input);
  return 0;
}
