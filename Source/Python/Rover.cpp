#include "Rover/Python/Rover.hpp"
#include <pybind11/pybind11.h>
#include "Rover/Python/LinearRegressionModel.hpp"
#include "Rover/Python/LogisticRegressionModel.hpp"
#include "Rover/Python/PolynomialRegressionModel.hpp"
#include "Rover/Python/SupportVectorMachine.hpp"

using namespace Eigen;
using namespace pybind11;
using namespace Rover;

PYBIND11_MODULE(rover, m) {
  export_linear_regression_model<double>(m, "");
  export_logistic_regression_model<double>(m, "");
  export_polynomial_regression_model<double>(m, "");
  export_support_vector_machine<double>(m, "");
  m.def("gradient_descent", [] (const object& f, int count) {
    return gradient_descent<double>([=] (const VectorX<double>& args) {
      return f(args).cast<VectorX<double>>();
    }, count);
  });
}
