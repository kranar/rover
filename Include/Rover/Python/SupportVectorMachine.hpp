#ifndef ROVER_PYTHON_SUPPORT_VECTOR_MACHINE_HPP
#define ROVER_PYTHON_SUPPORT_VECTOR_MACHINE_HPP
#include <string>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include "Rover/SupportVectorMachine.hpp"

namespace Rover {

  /**
   * Exports the generic SupportVectorMachine.
   * @param module The module to export to.
   * @param prefix The prefix to use when forming the type name.
   */
  template<typename S>
  void export_support_vector_machine(
      pybind11::module& module, const std::string& prefix) {
    using Model = SupportVectorMachine<S>;
    auto name = prefix + "SupportVectorMachine";
    if(pybind11::hasattr(module, name.c_str())) {
      return;
    }
    auto model = pybind11::class_<Model>(module, name.c_str())
      .def(pybind11::init<typename Model::Vector>())
      .def_property_readonly("parameters", &Model::get_parameters)
      .def("evaluate", &Model::evaluate);
    module.def(
      "train_support_vector_machine", &train_support_vector_machine<S>);
  }
}

#endif
