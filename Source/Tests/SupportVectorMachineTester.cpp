#include <doctest/doctest.h>
#include "Rover/SupportVectorMachine.hpp"

using namespace Eigen;
using namespace Rover;

TEST_SUITE("SupportVectorMachine") {
  TEST_CASE("construction") {
  }

  TEST_CASE("evaluate") {
  }

  TEST_CASE("zero") {
  }

  TEST_CASE("train") {
    auto sample = Eigen::MatrixXd(4, 3);
    sample << 1.0, 2.0, 1.0,
              2.0, 3.0, 1.0,
              -1.0, -2.0, -1.0,
              -2.0, -3.0, -1.0;
    auto model = train_support_vector_machine(sample);
    auto p1 = Eigen::VectorXd(2);
    p1 << 1.0, 2.0;
    REQUIRE(model.evaluate(p1));
    auto p2 = Eigen::VectorXd(2);
    p2 << 2.0, 3.0;
    REQUIRE(model.evaluate(p2));
    auto p3 = Eigen::VectorXd(2);
    p3 << -1.0, -2.0;
    REQUIRE(!model.evaluate(p3));
    auto p4 = Eigen::VectorXd(2);
    p4 << -2.0, -3.0;
    REQUIRE(!model.evaluate(p4));
  }

  TEST_CASE("train2") {
    auto sample = Eigen::MatrixXd(10, 3);
    sample << 1.0, 2.0, 1.0,
              2.0, 3.0, 1.0,
              3.0, 3.0, 1.0,
              4.0, 5.0, 1.0,
              5.0, 1.0, 1.0,
              7.0, 8.0, -1.0,
              8.0, 7.0, -1.0,
              9.0, 6.0, -1.0,
              6.0, 9.0, -1.0,
              8.0, 9.0, -1.0;
    auto model = train_support_vector_machine(sample);
  }
}
