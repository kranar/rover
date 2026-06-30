#ifndef ROVER_POLYNOMIAL_REGRESSION_MODEL_HPP
#define ROVER_POLYNOMIAL_REGRESSION_MODEL_HPP
#include <Eigen/Dense>

namespace Rover {

  /**
   * Implements a polynomial regression model.
   * @param <S> The type of scalar to use.
   */
  template<typename S>
  class PolynomialRegressionModel {
    public:

      /** The type of scalar to use. */
      using Scalar = S;

      /** The type used for the vector of parameters or point. */
      using Vector = Eigen::VectorX<Scalar>;

      /**
       * Constructs a PolynomialRegressionModel.
       * @param parameters The parameters to the polynomial regression model.
       * @param order The order of the polynomial.
       */
      PolynomialRegressionModel(Vector parameters, int order);

      /** Returns this model's parameters. */
      const Vector& get_parameters() const;

      /** Returns the order of the polynomial. */
      int get_order() const;

      /**
       * Evaluates this model at a point.
       * @param point The point to evaluate.
       * @return The evaluation of this model at the specified <i>point</i>.
       */
      Scalar evaluate(const Vector& point) const;

    private:
      Vector m_parameters;
      int m_order;
  };

  /**
   * Trains a PolynomialRegressionModel using regularized least squares.
   * @param sample The sample to train the model with.
   * @param order The order of the polynomial to fit.
   * @param lambda The regularization constant.
   * @return A polynomial regression model trained on the <i>sample</i>.
   */
  template<typename Scalar>
  PolynomialRegressionModel<Scalar> train_polynomial_regression_model(
    const Eigen::MatrixX<Scalar>& sample, int order,
    Scalar lambda = Scalar(0));
}

#endif
