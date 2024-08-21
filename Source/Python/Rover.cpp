#include "Rover/Python/Rover.hpp"
#include <pybind11/pybind11.h>
#include "Rover/Python/LinearRegressionModel.hpp"
#include "Rover/Python/LogisticRegressionModel.hpp"

using namespace pybind11;
using namespace Rover;

PYBIND11_MODULE(rover, m) {
  export_linear_regression_model<double>(m, "");
  export_logistic_regression_model<double>(m, "");
}
