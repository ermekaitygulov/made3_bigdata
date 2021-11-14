package linreg

import breeze.linalg.{DenseMatrix, inv}

case class LinearRegression() {
  private var theta: DenseMatrix[Double] = _

  def fit(featMatrix: DenseMatrix[Double], targetMatrix: DenseMatrix[Double]): DenseMatrix[Double] = {
    theta = inv.apply(featMatrix.t * featMatrix) * featMatrix.t * targetMatrix
    theta
  }

  def predict(featMatrix: DenseMatrix[Double]): DenseMatrix[Double] = {
    val prediction: DenseMatrix[Double] = featMatrix * theta
    prediction
  }
}
