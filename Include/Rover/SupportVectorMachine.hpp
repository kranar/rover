#ifndef ROVER_SUPPORT_VECTOR_MACHINE_HPP
#define ROVER_SUPPORT_VECTOR_MACHINE_HPP
#include <iostream>
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
    auto x_dot = points * points.transpose();
    auto y_dot = targets * targets.transpose();
    auto alpha =
      Eigen::VectorX<Scalar>(Eigen::VectorX<Scalar>::Zero(points.rows()));
    auto iterations = 10000;
    auto rate = 0.001;
    for(auto z = 0; z != iterations; ++z) {
      for(auto i = 0; i != points.rows(); ++i) {
        auto gradient = Scalar(1);
        for(auto j = 0; j != points.rows(); ++j) {
          gradient -= alpha(j) * y_dot(j) * x_dot(j);
        }
        alpha(i) = std::max(Scalar(0), alpha(i) + rate * gradient);
      }
      auto is_satisfied = false;
      while(!is_satisfied) {
        is_satisfied = true;
        auto error = Scalar(0);
        for(auto i = 0; i != points.rows(); ++i) {
          error += alpha(i) * targets(i);
        }
        auto tolerance = 0.0001;
        if(std::abs(error) > tolerance) {
          auto sum = alpha.sum();
          for(auto i = 0; i != alpha.rows(); ++i) {
            alpha(i) -= (alpha(i) * targets(i) * error / sum);
            if(alpha(i) < 0) {
              alpha(i) = 0;
              is_satisfied = false;
            }
          }
        }
      }
    }
/*
    auto K = points * points.transpose();
    auto A = Eigen::MatrixX<Scalar>(sample.rows() + 1, sample.rows() + 1);
    A.topLeftCorner(sample.rows(), sample.rows()) =
      K.cwiseProduct(targets * targets.transpose());
    A.topRightCorner(sample.rows(), 1) = targets;
    A.bottomLeftCorner(1, sample.rows()) = targets.transpose();
    A(sample.rows(), sample.rows()) = 0;
    auto b =
      Eigen::VectorX<Scalar>(Eigen::VectorX<Scalar>::Ones(sample.rows() + 1));
    b(sample.rows()) = 0;
    auto solution = Eigen::VectorX<Scalar>(A.colPivHouseholderQr().solve(b));
    auto alphas = solution.head(sample.rows());
    auto bias = solution(sample.rows());
    auto w = (alphas.cwiseProduct(targets).transpose() * points).transpose();
    auto parameters = Eigen::VectorX<Scalar>(sample.cols());
    parameters(0) = bias;
    parameters.tail(sample.cols() - 1) = w;
*/
    std::cout << alpha;
    auto parameters = Eigen::VectorX<Scalar>(sample.cols());
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
