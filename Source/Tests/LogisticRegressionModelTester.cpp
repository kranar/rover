#include <doctest/doctest.h>
#include "Rover/LogisticRegressionModel.hpp"

using namespace Eigen;
using namespace Rover;

TEST_SUITE("LogisticRegressionModel") {
  TEST_CASE("construction") {
    auto parameters = VectorXd(3);
    parameters << 0.5, -1.0, 2.0;
    auto model = LogisticRegressionModel(parameters);
    REQUIRE(model.get_parameters() == parameters);
  }

  TEST_CASE("evaluate") {
    auto parameters = VectorXd(3);
    parameters << 0.5, -1.0, 2.0;
    auto model = LogisticRegressionModel(parameters);
    auto regressors = VectorXd(3);
    regressors << 1.0, 2.0, -1.0;
    auto z = parameters.dot(regressors);
    auto expected = 1 / (1 + std::exp(-z));
    CHECK(
      model.evaluate(regressors) == doctest::Approx(expected).epsilon(1e-6));
  }

  TEST_CASE("zero") {
    auto parameters = VectorXd(VectorXd::Zero(3));
    auto model = LogisticRegressionModel(parameters);
    auto regressors = VectorXd(3);
    regressors << 1.0, 2.0, -1.0;
    CHECK(model.evaluate(regressors) == doctest::Approx(0.5).epsilon(1e-6));
  }

  TEST_CASE("train") {
    auto sample = MatrixX<double>(4, 3);
    sample << 1, 2, 0,
              2, 3, 0,
              4, 5, 1,
              5, 6, 1;
    auto model = train_logistic_regression_model(sample);
    auto regressors_0 = VectorXd(3);
    regressors_0 << 1, 2, 1;
    auto regressors_1 = VectorXd(3);
    regressors_1 << 4, 5, 1;
    CHECK(model.evaluate(regressors_0) < 0.5);
    CHECK(model.evaluate(regressors_1) > 0.5);
  }
}
