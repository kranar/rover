#include <doctest/doctest.h>
#include "Rover/LinearRegressionModel.hpp"

using namespace Eigen;
using namespace Rover;

TEST_SUITE("LinearRegressionModel") {
  TEST_CASE("construction") {
    auto parameters = VectorXd(2);
    parameters << 1.5, -0.5;
    auto model = LinearRegressionModel(parameters);
    REQUIRE(model.get_parameters() == parameters);
  }

  TEST_CASE("evaluate") {
    auto parameters = VectorXf(2);
    parameters << 2.0f, 3.0f;
    auto model = LinearRegressionModel(parameters);
    SUBCASE("x = 1") {
      auto regressors = VectorXf(1);
      regressors << 1.0f;
      auto result = model.evaluate(regressors);
      CHECK(result == doctest::Approx(5.0f));
    }
    SUBCASE("x = 0") {
      auto regressors = VectorXf(1);
      regressors << 0.0f;
      auto result = model.evaluate(regressors);
      CHECK(result == doctest::Approx(2.0f));
    }
  }

  TEST_CASE("train_zero_intercept") {
    auto sample = MatrixXf(3, 2);
    sample << 1, 1, 2, 2, 3, 3;
    auto model = train_linear_regression_model(sample);
    auto regressors = VectorXf(1);
    regressors << 5.0f;
    auto result = model.evaluate(regressors);
    CHECK(result == doctest::Approx(5.0f));
  }

  TEST_CASE("train_with_intercept") {
    auto sample = MatrixXf(3, 2);
    sample << 3, 4, 4, 5, 5, 6;
    auto model = train_linear_regression_model(sample);
    auto regressors = VectorXf(1);
    regressors << 9.0f;
    auto result = model.evaluate(regressors);
    CHECK(result == doctest::Approx(10.0f));
  }
}
