#ifndef ROVER_PERCEPTRON_HPP
#define ROVER_PERCEPTRON_HPP
#include <Eigen/Dense>

namespace Rover {

  /**
   * Implements a perceptron.
   * @param <S> The type of scalar to use.
   */
  template<typename S>
  class Perceptron {
    public:

      /** The type of scalar to use. */
      using Scalar = S;

      /** The type used for the vector of parameters and points. */
      using Vector = Eigen::VectorX<Scalar>;

      /**
       * Constructs a Perceptron.
       * @param parameters The parameters of the perceptron, with the bias term
       *        listed first.
       */
      explicit Perceptron(Vector parameters);

      /** Returns this model's parameters. */
      const Vector& get_parameters() const;

      bool evaluate(const Vector& point) const;

    private:
      Vector m_parameters;
  };

  template<typename Scalar>
  Perceptron<Scalar>::Perceptron(Vector parameters)
    : m_parameters(std::move(parameters)) {}

  template<typename Scalar>
  const typename Perceptron<Scalar>::Vector&
      Perceptron<Scalar>::get_parameters() const {
    return m_parameters;
  }

  template<typename Scalar>
  bool Perceptron<Scalar>::evaluate(const Vector& point) const {
    return m_parameters(0) +
      m_parameters.tail(m_parameters.size() - 1).dot(point) >= 0;
  }
}

#endif
