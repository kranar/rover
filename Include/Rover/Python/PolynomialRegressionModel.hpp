#ifndef ROVER_PYTHON_POLYNOMIAL_REGRESSION_MODEL_HPP
#define ROVER_PYTHON_POLYNOMIAL_REGRESSION_MODEL_HPP
#include <string>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include "Rover/PolynomialRegressionModel.hpp"

namespace Rover {

  /**
   * Exports the generic PolynomialRegressionModel.
   * @param module The module to export to.
   * @param prefix The prefix to use when forming the type name.
   */
  template<typename S>
  void export_polynomial_regression_model(
      pybind11::module& module, const std::string& prefix) {}
}

#endif
