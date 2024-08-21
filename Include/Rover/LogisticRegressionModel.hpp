#ifndef ROVER_LOGISTIC_REGRESSION_MODEL_HPP
#define ROVER_LOGISTIC_REGRESSION_MODEL_HPP
#include <cmath>
#include <Eigen/Dense>

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

      /** The type used for the vector of parameters or regressors. */
      using Vector = Eigen::VectorX<Scalar>;

      /**
       * Constructs a LogisticRegressionModel.
       * @param parameters The parameters to the linear regression model.
       */
      explicit LogisticRegressionModel(Vector parameters);

      /** Returns this model's parameters. */
      const Vector& get_parameters() const;

      /**
       * Evaluates this model on a list of regressors.
       * @param regressors The list of regressors to evaluate.
       * @return The evaluation of this model to the specified
       *         <i>regressor</i>s.
       */
      Scalar evaluate(const Vector& regressors) const;

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
    using Vector = typename LogisticRegressionModel<Scalar>::Vector;
    auto points = Eigen::MatrixX<Scalar>(sample.leftCols(sample.cols() - 1));
    auto targets = sample.rightCols(1);
    points.conservativeResize(points.rows(), points.cols() + 1);
    points.col(points.cols() - 1).setOnes();
    auto parameters = Vector(Vector::Zero(points.cols()));
    static const auto RATE = 0.001;
    static const auto ITERATIONS = 100000;
    auto sample_size = points.rows();
    for(auto i = 0; i < ITERATIONS; ++i) {
      auto predictions = (points * parameters).unaryExpr([] (auto z) {
        return static_cast<Scalar>(1) / (static_cast<Scalar>(1) + std::exp(-z));
      });
      parameters -=
        RATE * (points.transpose() * (predictions - targets) / sample_size);
    }
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
  LogisticRegressionModel<S>::Scalar
      LogisticRegressionModel<S>::evaluate(const Vector& regressors) const {
    auto z = m_parameters(1) +
      m_parameters.head(m_parameters.size() - 1).dot(regressors);
    return static_cast<Scalar>(1) / (static_cast<Scalar>(1) + std::exp(-z));
  }
}

#endif
