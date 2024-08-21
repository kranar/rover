#ifndef ROVER_PYTHON_LOGISTIC_REGRESSION_MODEL_HPP
#define ROVER_PYTHON_LOGISTIC_REGRESSION_MODEL_HPP
#include <string>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include "Rover/LogisticRegressionModel.hpp"

namespace Rover {

  /**
   * Exports the generic LogisticRegressionModel.
   * @param module The module to export to.
   * @param prefix The prefix to use when forming the type name.
   */
  template<typename S>
  void export_logistic_regression_model(
      pybind11::module& module, const std::string& prefix) {
    using Model = LogisticRegressionModel<S>;
    auto name = prefix + "LogisticRegressionModel";
    if(pybind11::hasattr(module, name.c_str())) {
      return;
    }
    auto model = pybind11::class_<Model>(module, name.c_str())
      .def(pybind11::init<typename Model::Vector>())
      .def_property_readonly("parameters", &Model::get_parameters)
      .def("evaluate", &Model::evaluate);
    module.def(
      "train_logistic_regression_model", &train_logistic_regression_model<S>);
  }
}

#endif
