#ifndef ROVER_OPTIMIZERS_HPP
#define ROVER_OPTIMIZERS_HPP
#include <Eigen/Dense>

namespace Rover {

  /**
   * Implements basic gradient descent to optimize a list of parameters.
   * @param f The function to optimize.
   * @param parameters The parameters of the function.
   * @param iterations The number of iterations to employ to find the optimum.
   * @param rate The learning rate.
   */
  template<typename Scalar, typename F>
  Eigen::VectorX<Scalar> gradient_descent(const F& f,
      Eigen::VectorX<Scalar> parameters, int iterations, Scalar rate) {
    for(auto i = 0; i < iterations; ++i) {
      auto gradient = f(parameters);
      parameters -= rate * gradient;
    }
    return parameters;
  }

  /**
   * Implements basic gradient descent to optimize a list of parameters using
   * 100000 iterations and a learning rate of 0.001.
   * @param f The function to optimize.
   * @param parameters The parameters of the function.
   */
  template<typename Scalar, typename F>
  Eigen::VectorX<Scalar> gradient_descent(
      const F& f, Eigen::VectorX<Scalar> parameters) {
    return gradient_descent(f, std::move(parameters), 100000, 0.001);
  }

  /**
   * Implements basic gradient descent to optimize a list of parameters using
   * 100000 iterations and a learning rate of 0.001.
   * @param f The function to optimize.
   */
  template<typename Scalar, typename F>
  Eigen::VectorX<Scalar> gradient_descent(const F& f, int parameter_count) {
    return gradient_descent(f,
      Eigen::VectorX<Scalar>(Eigen::VectorX<Scalar>::Zero(parameter_count)),
      100000, 0.001);
  }
}

#endif
