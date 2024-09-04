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
    auto targets = sample.col(sample.cols() - 1);
    auto x_dot = points * points.transpose();
    auto y_dot = targets * targets.transpose();
    auto alphas =
      Eigen::VectorX<Scalar>(Eigen::VectorX<Scalar>::Zero(points.rows()));
    auto b = Scalar(0);
    auto evaluator = [&] (auto i) {
      auto sum = b;
      for(auto j = 0; j != points.rows(); ++j) {
        sum += alphas(j) * targets(j) * x_dot(i, j);
      }
      return sum;
    };
    auto is_satisfied = false;
    while(!is_satisfied) {
      auto [a1, a2] = [&] {
        auto a1 = std::rand() % points.rows();
        auto a2 = std::rand() % points.rows();
        while(a2 == a1) {
          a2 = std::rand() % points.rows();
        }
        return std::tuple(a1, a2);
      }();
      static const auto THRESHOLD = 0.000001;
      auto n = x_dot(a1, a1) + x_dot(a2, a2) - 2 * x_dot(a1, a2); 
      if(std::abs(n) >= THRESHOLD) {
        auto zeta = targets.dot(alphas) - targets(a1) * alphas(a1) -
          targets(a2) - alphas(a2);
        auto e1 = evaluator(a1) - targets(a1);
        auto e2 = evaluator(a2) - targets(a2);
        auto next_a2 = alphas(a2) + (targets(a2) * (e1 - e2)) / n;
        if(targets(a1) != targets(a2)) {
          next_a2 = std::max(next_a2, std::max(0, alphas(a2) - alphas(a1)));
        }
        auto next_a1 =
          alphas(a1) + targets(a1) * targets(a2) * (alphas(a2) - next_a2);
        auto b1 = b - e1 - targets(a1) * (next_a1 - alphas(a1)) *
          x_dot(a1, a1) - targets(a2) * (next_a2 - alphas(a2)) * x_dot(a1, a2);
        auto b2 = b - e2 - targets(a1) * (next_a1 - alphas(a1)) *
          x_dot(a1, a2) - targets(a2) * (next_a2 - alphas(a2)) * x_dot(a2, a2);
        alphas(a1) = next_a1;
        alphas(a2) = next_a2;
        b = (b1 + b2) / 2;
      }
      is_satisfied = true;
      if(std::abs(alphas.dot(targets)) >= THRESHOLD) {
        is_satisfied = false;
      } else {
        for(auto i = 0; i != alphas.rows(); ++i) {
          if(alphas(i) > THRESHOLD) {
            if(std::abs(targets(i) * evaluator(i) - 1.0) > THRESHOLD) {
              is_satisfied = false;
              break;
            }
          } else if(targets(i) * evaluator(i) < THRESHOLD - 1.0) {
            is_satisfied = false;
            break;
          }
        }
      }
    }
    auto w = (alphas.cwiseProduct(targets).transpose() * points).transpose();
    auto parameters = Eigen::VectorX<Scalar>(sample.cols());
    parameters(0) = b;
    parameters.tail(sample.cols() - 1) = w;
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
