#ifndef ROVER_SUPPORT_VECTOR_MACHINE_HPP
#define ROVER_SUPPORT_VECTOR_MACHINE_HPP
#include <Eigen/Dense>

namespace Rover {

  /**
   * Implements a support vector machine to perform binary classification.
   * @param <S> The type of scalar to use.
   */
  template<typename Scalar>
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
}

#endif
