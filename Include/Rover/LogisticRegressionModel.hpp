#ifndef ROVER_LOGISTIC_REGRESSION_MODEL_HPP
#define ROVER_LOGISTIC_REGRESSION_MODEL_HPP
#include <cmath>
#include <utility>
#include <Eigen/Dense>
#include "Rover/Optimizers.hpp"

namespace Rover {

  /**
   * Implements a logistic regression model.
   * @param <S> The type of scalar to use.
   */
  template<typename S>
  class LogisticRegressionModel {
    public:

      /** The type of scalar to use. */
      using Scalar = S;

      /** The type used for the vector of parameters or point. */
      using Vector = Eigen::VectorX<Scalar>;

      /**
       * Constructs a LogisticRegressionModel.
       * @param parameters The parameters to the logistic regression model.
       */
      explicit LogisticRegressionModel(Vector parameters);

      /** Returns this model's parameters. */
      const Vector& get_parameters() const;

      /**
       * Evaluates this model at a point.
       * @param point The point to evaluate.
       * @return The evaluation of this model at the specified <i>point</i>.
       */
      Scalar evaluate(const Vector& point) const;

    private:
      Vector m_parameters;
  };

  /**
   * Trains a LogisticRegressionModel using log-loss minimization.
   * @param sample The sample to train the model with.
   * @return A logistic regression model trained on the <i>sample</i> using
   *         log-loss minimization.
   */
  template<typename Scalar>
  LogisticRegressionModel<Scalar> train_logistic_regression_model(
      const Eigen::MatrixX<Scalar>& sample) {
    auto points = Eigen::MatrixX<Scalar>(sample.leftCols(sample.cols() - 1));
    auto targets = sample.rightCols(1);
    points.conservativeResize(points.rows(), points.cols() + 1);
    points.col(points.cols() - 1).setOnes();
    auto parameters = gradient_descent<Scalar>([&] (const auto& parameters) {
      auto predictions = (points * parameters).unaryExpr([] (auto z) {
        return static_cast<Scalar>(1) / (static_cast<Scalar>(1) + std::exp(-z));
      });
      return (points.transpose() * (predictions - targets)) / points.rows();
    }, static_cast<int>(points.cols()));
    return LogisticRegressionModel(std::move(parameters));
  }

  template<typename S>
  LogisticRegressionModel<S>::LogisticRegressionModel(Vector parameters)
    : m_parameters(std::move(parameters)) {}

  template<typename S>
  const typename LogisticRegressionModel<S>::Vector&
      LogisticRegressionModel<S>::get_parameters() const {
    return m_parameters;
  }

  template<typename S>
  typename LogisticRegressionModel<S>::Scalar
      LogisticRegressionModel<S>::evaluate(const Vector& point) const {
    auto z = m_parameters(m_parameters.size() - 1) +
      m_parameters.head(m_parameters.size() - 1).dot(point);
    return static_cast<Scalar>(1) / (static_cast<Scalar>(1) + std::exp(-z));
  }
}

#endif
