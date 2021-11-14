import breeze.linalg._
import breeze.linalg.inv
import breeze.numerics.pow
import breeze.stats.mean

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

val projPath: String = "/home/airan/Projects/made/made3_bigdata/HW3"
val dataPath: String = projPath + "/insurance.csv"


def binValue(value: String, bin: String): Int = {
  if (value == bin) {
    1
  } else {
    0
  }
}

def readInsuranceDataset(path: String) = {
  val bufferedSource = Source.fromFile(path)
  var step = 0
  val feature = new ArrayBuffer[Array[Double]]()
  val target = new ArrayBuffer[Double]()
  for (line <- bufferedSource.getLines) {
    if (step > 0) {
      val cols = line.split(",").map(_.trim)
      // do whatever you want with the columns here
      val age = cols(0).toDouble
      val sex = binValue(cols(1),"male")
      val bmi = cols(2).toDouble
      val children = cols(3).toDouble
      val smoker = binValue(cols(4), "yes")
      val charges = cols(6).toDouble
      val row: Array[Double] = Array(1.0F, age, sex, bmi, children, smoker)
      feature += row
      target += charges
    }
    step += 1
  }
  bufferedSource.close
  println(s"${feature.size}")
  val featArray = feature.toArray
  val targetArray = target.toArray
  (featArray, targetArray)
}

case class LinearRegression() {
  private var theta: DenseMatrix[Double] = _

  def fit(featMatrix: DenseMatrix[Double], targetMatrix: DenseMatrix[Double]) = {
    theta = inv.apply(featMatrix.t * featMatrix) * featMatrix.t * targetMatrix
    theta
  }: DenseMatrix[Double]

  def predict(featMatrix: DenseMatrix[Double]) = {
    val prediction: DenseMatrix[Double] = featMatrix * theta
    prediction
  }: DenseMatrix[Double]
}

def rmseScore(trueTarget: DenseMatrix[Double], predTarget: DenseMatrix[Double],
              squared: Boolean = false) = {
  val mseElementWise: DenseMatrix[Double] = pow(trueTarget - predTarget, 2)
  val meanSquaredError: Double = mean(mseElementWise)
  if (squared) {
    meanSquaredError
  } else {
    val rmse: Double = pow(meanSquaredError, 0.5)
    rmse
  }
}: Double

def r2Score(trueTarget: DenseMatrix[Double], predTarget: DenseMatrix[Double]) = {
  val mse = rmseScore(trueTarget, predTarget, squared = true)
  val meanTarget = DenseMatrix.fill(trueTarget.rows, 1){mean(trueTarget)}
  val variance = rmseScore(trueTarget, meanTarget, squared = true)
  val r2 = 1 - (mse / variance)
  r2
}

val (featArray, targetArray) = readInsuranceDataset(dataPath)
val featMatrix: DenseMatrix[Double] = DenseMatrix(featArray:_*)
val targetMatrix: DenseMatrix[Double] = DenseMatrix(targetArray:_*)

val model = LinearRegression()
model fit (featMatrix, targetMatrix)
val prediction = model predict featMatrix
rmseScore(targetMatrix, prediction)
r2Score(targetMatrix, prediction)