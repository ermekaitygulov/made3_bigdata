import breeze.linalg.DenseMatrix
import breeze.numerics.pow
import breeze.stats.mean

package object linreg {
  def rmseScore(trueTarget: DenseMatrix[Double], predTarget: DenseMatrix[Double],
                squared: Boolean = false): Double = {
    val mseElementWise: DenseMatrix[Double] = pow(trueTarget - predTarget, 2)
    val meanSquaredError: Double = mean(mseElementWise)
    if (squared) {
      meanSquaredError
    } else {
      val rmse: Double = pow(meanSquaredError, 0.5)
      rmse
    }
  }

  def r2Score(trueTarget: DenseMatrix[Double], predTarget: DenseMatrix[Double]): Double = {
    val mse = rmseScore(trueTarget, predTarget, squared = true)
    val meanTarget = DenseMatrix.fill(trueTarget.rows, 1){mean(trueTarget)}
    val variance = rmseScore(trueTarget, meanTarget, squared = true)
    val r2 = 1 - (mse / variance)
    r2
  }
}
