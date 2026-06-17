#include <doctest/doctest.h>
#include "Rover/SupportVectorMachine.hpp"

using namespace Eigen;
using namespace Rover;

TEST_SUITE("SupportVectorMachine") {
  TEST_CASE("construction") {
    auto parameters = VectorXd(3);
    parameters << -3.8, 0.8, 0.4;
    auto model = SupportVectorMachine(parameters);
    REQUIRE(model.get_parameters() == parameters);
  }

  TEST_CASE("evaluate") {
    auto parameters = VectorXd(3);
    parameters << -3.8, 0.8, 0.4;
    auto model = SupportVectorMachine(parameters);
    auto positive = VectorXd(2);
    positive << 4.0, 4.0;
    REQUIRE(model.evaluate(positive));
    auto negative = VectorXd(2);
    negative << 1.0, 1.0;
    REQUIRE(!model.evaluate(negative));
  }

  TEST_CASE("zero") {
    auto parameters = VectorXd(VectorXd::Zero(3));
    auto model = SupportVectorMachine(parameters);
    auto regressors = VectorXd(2);
    regressors << 1.0, 2.0;
    REQUIRE(model.evaluate(regressors));
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
    auto sample = Eigen::MatrixXd(4, 3);
    sample << 1.0, 1.0, -1.0,
              2.0, 3.0, -1.0,
              4.0, 4.0, 1.0,
              5.0, 2.0, 1.0;
    auto model = train_support_vector_machine(sample);
    REQUIRE(model.get_parameters()[0] == doctest::Approx(-3.8));
    REQUIRE(model.get_parameters()[1] == doctest::Approx(0.8));
    REQUIRE(model.get_parameters()[2] == doctest::Approx(0.4));
  }
}
