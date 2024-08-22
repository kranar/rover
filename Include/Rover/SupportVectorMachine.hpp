#ifndef ROVER_SUPPORT_VECTOR_MACHINE_HPP
#define ROVER_SUPPORT_VECTOR_MACHINE_HPP
#include <Eigen/Dense>
#include "Rover/Optimizers.hpp"

namespace Rover {

  /**
   * Implements a support vector machine to perform binary classification.
   * @param <S> The type of scalar to use.
   */
  template<typename S>
  class SupportVectorMachine {
    public:

      /** The type of scalar to use. */
      using Scalar = S;

      /** The type used for the vector of parameters and points. */
      using Vector = Eigen::VectorX<Scalar>;

      /**
       * Constructs a SupportVectorMachine.
       * @param parameters The parameters of the line separating the two
       *        classes.
       */
      explicit SupportVectorMachine(Vector parameters);

      /** Returns this model's parameters. */
      const Vector& get_parameters() const;

      bool evaluate(const Vector& point) const;

    private:
      Vector m_parameters;
  };

  /**
   * Trains a SupportVectorMachine.
   * @param sample The sample to train the model with.
   * @return A support vector machine trained on the <i>sample</i>.
   */
  template<typename Scalar>
  SupportVectorMachine<Scalar> train_support_vector_machine(
      const Eigen::MatrixX<Scalar>& sample) {
    auto points = sample.leftCols(sample.cols() - 1);
    auto targets = sample.rightCols(1);
    auto parameters =
      Eigen::VectorX<Scalar>(Eigen::VectorX<Scalar>::Zero(points.cols() + 1));
    auto learning_rate = Scalar(0.01);
    auto max_iterations = 1000;
    for(int i = 0; i < max_iterations; ++i) {
      for(int j = 0; j < points.rows(); ++j) {
        auto x_i = points.row(j).transpose();
        auto y_i = targets(j, 0);
        auto extended_point = Eigen::VectorX<Scalar>(x_i.size() + 1);
        extended_point << 1, x_i;
        if(y_i * (parameters.dot(extended_point)) < 1) {
          parameters += learning_rate * y_i * extended_point;
        }
      }
      parameters.tail(points.cols()) -=
        learning_rate * parameters.tail(points.cols());
    }
    return SupportVectorMachine(std::move(parameters));
  }

  template<typename Scalar>
  SupportVectorMachine<Scalar>::SupportVectorMachine(Vector parameters)
    : m_parameters(std::move(parameters)) {}

  template<typename Scalar>
  const typename SupportVectorMachine<Scalar>::Vector&
      SupportVectorMachine<Scalar>::get_parameters() const {
    return m_parameters;
  }

  template<typename Scalar>
  bool SupportVectorMachine<Scalar>::evaluate(const Vector& point) const {
    return m_parameters(0) +
      m_parameters.tail(m_parameters.size() - 1).dot(point) >= 0;
  }
}

#endif
